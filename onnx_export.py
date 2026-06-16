# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import json
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
_ATTN_KEYWORDS = frozenset({"to_q", "to_k", "to_v", "to_out", "to_qkv", "to_logits", "token_emb"})


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


def get_ff_matmul_nodes(onnx_path):
    model = onnx.load(onnx_path)
    ff_nodes = []
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        name = node.name.lower()
        if any(kw in name for kw in _ATTN_KEYWORDS):
            continue
        ff_nodes.append(node.name)
    return ff_nodes


def quantize_onnx(input_path, output_path):
    nodes = get_ff_matmul_nodes(input_path)
    print(f"Quantizing {len(nodes)} FeedForward MatMul nodes in {input_path}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=nodes
    )
    print(f"Quantized model saved: {output_path}")


def save_model_config(save_dir, config):
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
    save_model_config(save_dir, config)

    quantize_onnx(f"{save_dir}/encoder_fp32.onnx", f"{save_dir}/encoder_int8.onnx")
    quantize_onnx(f"{save_dir}/decoder_fp32.onnx", f"{save_dir}/decoder_int8.onnx")