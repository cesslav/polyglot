# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import json
import shutil
import tempfile
import torch
import os
import onnx
from imp import Transformer
from onnxruntime.quantization import quantize_dynamic, QuantType


SRC_LANG = "RU"
TGT_LANG = "EN"
BIDIRECTIONAL = True
MODEL_VERSION = "0.1"
ARCH_VERSION = "v9"
PAD_ID = 3

QUANT_CONFIG = {
    "attention": "fp32",
    "feedforward": "int8",
    "logits": "fp16",
    "embeddings": "fp16",
}

PER_CHANNEL_INT8 = True

_FEEDFORWARD_KEYWORDS = frozenset({"w_gate", "w_up", "w_down"})
_LOGITS_KEYWORDS = frozenset({"to_logits", "head"})
_EMBEDDING_KEYWORDS = frozenset({"token_emb"})


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(src, mask=src_mask)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.head = model.to_logits

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.decoder(tgt, memory, context_mask=src_mask))


def export_fp32(model, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model = model.to("cpu")

    dummy_src = torch.randint(0, config["vocab_size"], (2, 16))
    dummy_src_mask = dummy_src.ne(PAD_ID)

    encoder_w = EncoderWrapper(model.encoder)
    torch.onnx.export(
        encoder_w,
        (dummy_src, dummy_src_mask),
        f"{save_dir}/encoder_fp32.onnx",
        input_names=["src", "src_mask"],
        output_names=["memory"],
        dynamic_axes={
            "src": {0: "batch", 1: "seq"},
            "src_mask": {0: "batch", 1: "seq"},
            "memory": {0: "batch", 1: "seq"},
        },
        dynamo=False
    )

    decoder_w = DecoderWrapper(model)
    dummy_tgt = torch.randint(0, config["vocab_size"], (2, 8))
    dummy_memory = torch.randn(2, 16, config["d_model"])
    dummy_src_mask_dec = torch.ones(2, 16, dtype=torch.bool)

    torch.onnx.export(
        decoder_w,
        (dummy_tgt, dummy_memory, dummy_src_mask_dec),
        f"{save_dir}/decoder_fp32.onnx",
        input_names=["tgt", "memory", "src_mask"],
        output_names=["logits"],
        dynamic_axes={
            "tgt": {0: "batch", 1: "tgt_seq"},
            "memory": {0: "batch", 1: "src_seq"},
            "src_mask": {0: "batch", 1: "src_seq"},
            "logits": {0: "batch", 1: "tgt_seq"},
        },
        dynamo=False
    )
    print("FP32 export done")


def classify_nodes(onnx_path):
    model = onnx.load(onnx_path)
    groups = {"attention": [], "logits": [], "embeddings": [], "feedforward": []}
    for node in model.graph.node:
        name = node.name.lower()
        if node.op_type in ("MatMul", "Gemm"):
            if any(kw in name for kw in _LOGITS_KEYWORDS):
                groups["logits"].append(node.name)
            elif any(kw in name for kw in _FEEDFORWARD_KEYWORDS):
                groups["feedforward"].append(node.name)
            else:
                groups["attention"].append(node.name)
        elif node.op_type == "Gather" and any(kw in name for kw in _EMBEDDING_KEYWORDS):
            groups["embeddings"].append(node.name)

    for group, nodes in groups.items():
        print(f"{group}: найдено {len(nodes)} узлов ({onnx_path})")
    return groups


def quantize_int8(input_path, output_path, nodes_to_quantize, per_channel=PER_CHANNEL_INT8):
    print(f"INT8: квантизация {len(nodes_to_quantize)} узлов ({input_path})")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=nodes_to_quantize,
        per_channel=per_channel,
    )
    print(f"INT8 модель сохранена: {output_path}")


def quantize_fp16(input_path, output_path, target_nodes):
    try:
        from onnxconverter_common import float16
    except ImportError as e:
        raise ImportError("Не найден onnxconverter_common: pip install onnxconverter-common") from e

    model = onnx.load(input_path)
    all_names = {node.name for node in model.graph.node if node.name}
    block_list = list(all_names - set(target_nodes))

    fp16_model = float16.convert_float_to_float16(
        model,
        node_block_list=block_list,
        keep_io_types=True,
        disable_shape_infer=True,
    )
    onnx.save_model(fp16_model, output_path)
    print(f"FP16: {len(target_nodes)} узлов переведено в half precision -> {output_path}")


def apply_quantization_config(fp32_path, output_path, config, tmp_dir):
    groups = classify_nodes(fp32_path)
    current_path = fp32_path

    int8_groups = [g for g, p in config.items() if p == "int8" and groups.get(g)]
    if int8_groups:
        nodes_to_quantize = [n for g in int8_groups for n in groups[g]]
        step_path = os.path.join(tmp_dir, "step_int8.onnx")
        quantize_int8(current_path, step_path, nodes_to_quantize=nodes_to_quantize)
        current_path = step_path

    fp16_groups = [g for g, p in config.items() if p == "fp16" and groups.get(g)]
    if fp16_groups:
        target_nodes = [n for g in fp16_groups for n in groups[g]]
        step_path = os.path.join(tmp_dir, "step_fp16.onnx")
        quantize_fp16(current_path, step_path, target_nodes=target_nodes)
        current_path = step_path

    shutil.copy(current_path, output_path)
    print(f"Готово: {output_path}")


def save_model_config(save_dir, config, quant_config):
    meta = {
        "input_language": SRC_LANG,
        "output_language": TGT_LANG,
        "bidirectional": BIDIRECTIONAL,
        "model_version": MODEL_VERSION,
        "arch": ARCH_VERSION,
        "d_model": config["d_model"],
        "vocab_size": config["vocab_size"],
        "num_layers": config["num_layers"],
        "dim_head": config["dim_head"],
        "num_heads": config["num_heads"],
        "mlp_mult": config["mlp_mult"],
        "quantization": quant_config,
    }
    path = os.path.join(save_dir, "model_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"model_config.json saved: {path}")


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    config = checkpoint["config"]

    model = Transformer(
        dim=config["d_model"],
        enc_num_tokens=config["vocab_size"],
        enc_depth=config["num_layers"],
        enc_heads=config["num_heads"],
        enc_dim_head=config["dim_head"],
        enc_mlp_mult=config["mlp_mult"],
        dec_num_tokens=config["vocab_size"],
        dec_depth=config["num_layers"] + config["dec_depth_diff"],
        dec_heads=config["num_heads"],
        dec_dim_head=config["dim_head"],
        dec_mlp_mult=config["mlp_mult"],
        dropout=config["dropout"],
        tie_token_emb=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

    save_dir = "onnx_export"
    checkpoint = "./transformer.pt"

    model, config = load_model(checkpoint)

    export_fp32(model, config, save_dir)
    save_model_config(save_dir, config, QUANT_CONFIG)

    with tempfile.TemporaryDirectory() as tmp_dir:
        apply_quantization_config(
            f"{save_dir}/encoder_fp32.onnx", f"{save_dir}/encoder_quant.onnx", QUANT_CONFIG, tmp_dir
        )
        apply_quantization_config(
            f"{save_dir}/decoder_fp32.onnx", f"{save_dir}/decoder_quant.onnx", QUANT_CONFIG, tmp_dir
        )