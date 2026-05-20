# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import sys
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from torch.utils.data import DataLoader
from imp import Transformer
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from collections import deque


def get_transformer_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        step = max(step, 1)
        if step < warmup_steps:
            return step / warmup_steps

        return (warmup_steps ** 0.65) / (step ** 0.65)

    return LambdaLR(optimizer, lr_lambda)


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


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        if m.bias is not None:
            nn.init.constant_(m.weight, 1.0)


def save(transformer, epoch, optimizer, scheduler, train_loss=0, val_loss="NaN", progress=0):
    global last_save, config, cold_save_counter, batch_size
    os.makedirs("./cold_saves/", exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'progress': progress,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'time': datetime.now(),
        'config': config
    }

    torch.save(checkpoint, os.path.join(checkpoint_dir, f"transformer_epoch_{epoch}.pt"))
    cold_save_counter += 1
    if cold_save_counter % 5 == 0:
        torch.save(checkpoint, os.path.join("./cold_saves/", f"transformer_epoch_{epoch}_{config["d_model"]}{config["num_layers"]}{config["mlp_mult"]}_{train_loss:.3f}.pt"))
    last_save = datetime.now()


def train_epoch(model, loader, optimizer, scheduler, criterion, device, num, accumulation_steps=6):
    model.train()
    total_loss = 0.0
    accum_loss = 0.0
    last_thousand_loss = deque(maxlen=1000)
    last_thousand_sum = 0.0
    counter = 0
    step = 0
    skipped = 0
    loop = tqdm(loader, desc=f"Epoch {num}")

    for batch in loop:
        src = batch["input"].to(device, non_blocking=True).squeeze()
        tgt = batch["output"].to(device, non_blocking=True).squeeze()

        try:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output_fwd = model(src, tgt[:, :-1])
                loss_fwd = criterion(
                    output_fwd.reshape(-1, vocab_size),
                    tgt[:, 1:].reshape(-1)
                ) / accumulation_steps

            loss_fwd.backward()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output_bwd = model(tgt, src[:, :-1])
                loss_bwd = criterion(
                    output_bwd.reshape(-1, vocab_size),
                    src[:, 1:].reshape(-1)
                ) / accumulation_steps

            loss_bwd.backward()

        except RuntimeError as e:
            skipped += 1
            accum_loss = 0.0
            optimizer.zero_grad()
            print(f"Пропуск батча: ошибка backward: {e}, пропусков: {skipped}")
            continue

        batch_loss = loss_fwd.item() + loss_bwd.item()
        accum_loss += batch_loss
        step += 1

        if step % accumulation_steps == 0 or step == len(loader):
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += accum_loss
            counter += 1

            if len(last_thousand_loss) == last_thousand_loss.maxlen:
                last_thousand_sum -= last_thousand_loss[0]
            last_thousand_loss.append(accum_loss)
            last_thousand_sum += accum_loss


            lrs = [group["lr"] for group in optimizer.param_groups]

            min_lr = min(lrs)
            max_lr = max(lrs)
            avg_lr = sum(lrs) / len(lrs)

            with torch.no_grad():
                total_param_norm = 0.0

                for p in model.parameters():
                    if p.requires_grad:
                        total_param_norm += p.norm().item() ** 2

                total_param_norm = total_param_norm ** 0.5

            update_ratio = total_norm / total_param_norm

            loop.set_postfix_str(
                f"loss: {accum_loss:.6f}  "
                f"avg: {total_loss / counter:.6f}  "
                f"avg1k: {last_thousand_sum / len(last_thousand_loss):.6f}  "
                f"max1k: {max(last_thousand_loss):.6f}  "
                f"step: {counter} "
                f"norm: {total_norm:.6f}  "
                f"save: {datetime.now() - last_save} "
                f"skipped_steps: {skipped} "
                f"lr_min: {min_lr:.4e}  "
                f"lr_max: {max_lr:.4e}  "
                f"lr_avg: {avg_lr:.4e}  "
                f"upd_ratio: {update_ratio:.4e}"
            )
            accum_loss = 0.0

            if datetime.now() - last_save > timedelta(hours=1):
                save(
                    model, num, optimizer, scheduler,
                    train_loss=(last_thousand_sum / len(last_thousand_loss)),
                    progress=(loop.format_dict["n"] / loop.format_dict["total"])
                )

    return total_loss / counter


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    loop = tqdm(loader)
    with torch.no_grad():
        for batch in loop:
            src, tgt = batch["input"].to(device).squeeze(), batch["output"].to(device).squeeze()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

    torch.set_float32_matmul_precision('high')
    train_config = json.load(open("train_config.json", mode="r"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "d_model": 256,
        "vocab_size": 48000,
        "num_layers": 4,
        "dec_depth_diff": 0,
        "dim_head": 32,
        "num_heads": 0,
        "mlp_mult": 4,
        "dropout": 0,
    }

    num_epochs = train_config["max_epochs"]
    batch_size = train_config["batch_size"][-1]
    is_continue = train_config["continue"]
    checkpoint_dir = train_config["save_dir"]
    checkpoint_name = train_config["checkpoint"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(os.path.join(checkpoint_dir, checkpoint_name))

    if is_continue:
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
        if not checkpoint_name:
            print("Указано пустое имя файла при продолжении обучения! Запуск с нуля.")
            is_continue = False
        elif not os.path.isfile(ckpt_path):
            print(f"Файл контрольной точки не найден: {ckpt_path}. Запуск с нуля.")
            is_continue = False
        else:
            print(f"Чекпоинт: {ckpt_path}")

    if not is_continue:
        if config["num_heads"] == 0:
            config["num_heads"] = config["d_model"] // config["dim_head"]

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
            tie_token_emb=True,
        ).to(device)

        param_groups = get_transformer_lrd(transformer, base_lr=train_config["init_lr"],
                                           decay=train_config["lr_decay"], weight_decay=0.01)
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-9, fused=True)
        scheduler = get_transformer_scheduler(optimizer, warmup_steps=32000)
        transformer.apply(init_weights)
        start_epoch = 0
        progress = 0
        vocab_size = config["vocab_size"]

    else:
        ckpt = torch.load(
            os.path.join(checkpoint_dir, checkpoint_name),
            weights_only=False,
            map_location=device,
        )
        config = ckpt["config"]

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
            tie_token_emb=True,
        ).to(device)

        param_groups = get_transformer_lrd(transformer, base_lr=1, decay=1, weight_decay=0.01)
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-9, fused=True)
        scheduler = get_transformer_scheduler(optimizer, warmup_steps=8000)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        transformer.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("Контрольная точка успешно загружена!")
        progress = ckpt["progress"]
        train_loss = ckpt["train_loss"]
        start_epoch = ckpt["epoch"] - 1
        vocab_size = config["vocab_size"]

    print("model initialized!")
    print(config)
    print(sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9)
    print(torch.cuda.memory_allocated() / 1024 ** 3)

    try:
        train_loader = DataLoader(
            load_from_disk(train_config["train_ds_dir"]),
            batch_size=batch_size, shuffle=True, num_workers=12,
            pin_memory=True, drop_last=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            load_from_disk(train_config["val_ds_dir"]),
            batch_size=batch_size, num_workers=12,
            pin_memory=True, persistent_workers=True,
        )
    except Exception as e:
        print(e)
        sys.exit(1)

    criterion = nn.CrossEntropyLoss(ignore_index=3, label_smoothing=0.05)

    cold_save_counter = 0
    time.sleep(0.5)
    last_save = datetime.now()

    for epoch in range(start_epoch + 1, num_epochs + 1):
        if progress < 0.9:
            train_loss = train_epoch(transformer, train_loader, optimizer, scheduler, criterion, device, epoch,
                                     accumulation_steps=192 // batch_size)
        progress = 0
        val_loss = evaluate(transformer, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(torch.cuda.memory_allocated() / 1024 ** 3)

        save(transformer, epoch, optimizer, scheduler, train_loss, val_loss)
