import os

import torch
from torch import optim

from imp import Transformer


def get_transformer_lrd(model, base_lr=1e-4, decay=0.9, weight_decay=0.01):
    encoder_layers = len(model.encoder.layer)
    decoder_layers = len(model.decoder.layer)
    max_depth = encoder_layers + decoder_layers + 1

    no_decay = {"bias", "gamma", "beta"}

    layer_map = {}
    used = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in used:
            continue
        used.add(id(param))

        if "token_emb" in name:
            depth = 0
        elif "encoder.layer" in name:
            idx = int(name.split("encoder.layer.")[1].split(".")[0])
            depth = idx + 1
        elif "decoder.layer" in name:
            idx = int(name.split("decoder.layer.")[1].split(".")[0])
            depth = encoder_layers + idx + 1
        else:
            depth = max_depth

        if depth not in layer_map:
            layer_map[depth] = {"decay": [], "no_decay": []}

        is_no_decay = any(nd in name for nd in no_decay)
        bucket = "no_decay" if is_no_decay else "decay"
        layer_map[depth][bucket].append(param)

    param_groups = []
    for depth, buckets in layer_map.items():
        lr = base_lr * (decay ** (max_depth - depth))

        if buckets["decay"]:
            param_groups.append({
                "params": buckets["decay"],
                "lr": lr,
                "weight_decay": weight_decay
            })

        if buckets["no_decay"]:
            param_groups.append({
                "params": buckets["no_decay"],
                "lr": lr,
                "weight_decay": 0.0
            })

    return param_groups


checkpoint = torch.load("./t5s/transformer_epoch_1.pt", weights_only=False, map_location="cpu")
config = checkpoint["config"]
transformer = Transformer(
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
        ).to("cpu")

param_groups = get_transformer_lrd(
    transformer,
    base_lr=0.000003,
    decay=0.9,
    weight_decay=0.01
)
optimizer = optim.AdamW(
    param_groups,
    betas=(0.9, 0.98),
    eps=1e-9
)

print(config)
# print(checkpoint[""])
print(checkpoint["time"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
for group in optimizer.param_groups:
    print(group["lr"])



# Epoch 1:   1%|▏         | 83140/6330891 [9:56:46<748:33:27,  2.32it/s, loss: 4.672296  avg: 4.540804|4.652693  avg1k: 4.574516|4.678251  max1k: 5.880175|6.077821  step: 83141  norm: 19.390278  save: 0:14:48.130618  skipped: 0  phase: fwd]
# Epoch 1:   0%|          | 3705/6330891 [24:33<702:49:18,  2.50it/s, loss: 13.734193  avg: 13.755313  avg1k: 13.755313  max1k: 15.406622  step: 463 norm: 21.618549  save: 0:24:32.719004 skipped_steps: 0]
