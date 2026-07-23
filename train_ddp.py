# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import sys
import json
import copy
import time
import math
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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
    encoder_layers = len(model.encoder.layers)
    decoder_layers = len(model.decoder.layers)
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
            depth = max_depth
        elif "encoder.layers" in name:
            idx = int(name.split("encoder.layers.")[1].split(".")[0])
            depth = idx + 1
        elif "decoder.layers" in name:
            idx = int(name.split("decoder.layers.")[1].split(".")[0])
            depth = encoder_layers + idx + 1
        else:
            depth = max_depth

        if depth not in layer_map:
            layer_map[depth] = {"decay": [], "no_decay": []}

        bucket = "no_decay" if any(nd in name for nd in no_decay) else "decay"
        layer_map[depth][bucket].append(param)

    param_groups = []
    for depth, buckets in layer_map.items():
        lr = base_lr * (decay ** (max_depth - depth))
        if buckets["decay"]:
            param_groups.append({"params": buckets["decay"], "lr": lr, "weight_decay": weight_decay})
        if buckets["no_decay"]:
            param_groups.append({"params": buckets["no_decay"], "lr": lr, "weight_decay": 0.0})

    return param_groups


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


def save(transformer, epoch, optimizer, scheduler, train_loss=0, val_loss="NaN", progress=0):
    global last_save, config, cold_save_counter
    os.makedirs("./cold_saves/", exist_ok=True)
    if rank != 0:
        last_save = datetime.now()
        return

    model_state = (transformer.module.state_dict()
                   if hasattr(transformer, "module")
                   else transformer.state_dict())

    checkpoint = {
        'epoch': epoch,
        'progress': progress,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'time': datetime.now(),
        'config': config,
    }

    torch.save(checkpoint, os.path.join(checkpoint_dir, f"transformer_epoch_{epoch}.pt"))
    cold_save_counter += 1
    if cold_save_counter % 5 == 0:
        os.makedirs("./cold_saves/", exist_ok=True)
        torch.save(checkpoint, os.path.join(
            "./cold_saves/",
            f"transformer_epoch_{epoch}_{config['d_model']}{config['num_layers']}{config['mlp_mult']}_{train_loss:.3f}.pt"
        ))
    last_save = datetime.now()


def snapshot_state(model, optimizer):
    module = model.module if hasattr(model, "module") else model
    model_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def restore_state(model, optimizer, snapshot):
    model_state, optimizer_state = snapshot
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, num,
                 accumulation_steps=12, clip_window=30, clip_mult=1.25, clip_default=3,
                 snapshot_interval=100):
    model.train()
    total_loss = 0.0
    accum_loss = 0.0
    last_thousand_loss = deque(maxlen=1000)
    last_thousand_sum = 0.0
    grad_norm_history = deque(maxlen=clip_window)
    grad_norm_sum = 0.0
    counter = 0
    step = 0
    skipped = 0
    rolled_back = 0
    last_good = snapshot_state(model, optimizer)

    loop = tqdm(loader, desc=f"Epoch {num}") if rank == 0 else loader

    for batch in loop:
        src = batch["input"].to(device, non_blocking=True).squeeze()
        tgt = batch["output"].to(device, non_blocking=True).squeeze()

        try:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output_fwd = model(src, tgt[:, :-1], src_mask=src.ne(PAD_ID), tgt_mask=tgt[:, :-1].ne(PAD_ID))
                loss_fwd = criterion(
                    output_fwd.reshape(-1, vocab_size),
                    tgt[:, 1:].reshape(-1)
                ) / accumulation_steps * grad_scale
            with model.no_sync():
                loss_fwd.backward()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output_bwd = model(tgt, src[:, :-1], src_mask=tgt.ne(PAD_ID), tgt_mask=src[:, :-1].ne(PAD_ID))
                loss_bwd = criterion(
                    output_bwd.reshape(-1, vocab_size),
                    src[:, 1:].reshape(-1)
                ) / accumulation_steps * grad_scale
            if step % accumulation_steps != accumulation_steps - 1:
                with model.no_sync():
                    loss_bwd.backward()
            else:
                loss_bwd.backward()

        except RuntimeError as e:
            skipped += 1
            accum_loss = 0.0
            optimizer.zero_grad()
            if rank == 0:
                print(f"Пропуск батча: ошибка backward: {e}, пропусков: {skipped}")
            continue

        batch_loss = loss_fwd.item() + loss_bwd.item()
        accum_loss += batch_loss
        step += 1

        if step % accumulation_steps == 0 or step == len(loader):
            clip_norm = (
                grad_norm_sum / len(grad_norm_history) * clip_mult
                if len(grad_norm_history) == clip_window else clip_default
            )
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            if not torch.isfinite(total_norm):
                skipped += 1
                optimizer.zero_grad()
                accum_loss = 0.0
                if rank == 0:
                    print(f"Пропуск шага: total_norm={total_norm}, пропусков: {skipped}")
                continue

            if len(grad_norm_history) == clip_window:
                grad_norm_sum -= grad_norm_history[0]
            grad_norm_history.append(total_norm.item())
            grad_norm_sum += total_norm.item()

            optimizer.step()

            with torch.no_grad():
                total_param_norm = sum(
                    p.norm().item() ** 2
                    for p in model.parameters() if p.requires_grad
                ) ** 0.5

            if not math.isfinite(total_param_norm):
                rolled_back += 1
                restore_state(model, optimizer, last_good)
                optimizer.zero_grad()
                accum_loss = 0.0
                if rank == 0:
                    print(f"Обнаружен NaN в весах, откат на последний снимок.")
                continue

            scheduler.step()
            optimizer.zero_grad()

            total_loss += accum_loss
            counter += 1

            if len(last_thousand_loss) == last_thousand_loss.maxlen:
                last_thousand_sum -= last_thousand_loss[0]
            last_thousand_loss.append(accum_loss)
            last_thousand_sum += accum_loss

            if counter % snapshot_interval == 0:
                last_good = snapshot_state(model, optimizer)

            if rank == 0:
                lrs = [g["lr"] for g in optimizer.param_groups]
                min_lr = min(lrs)
                max_lr = max(lrs)
                avg_lr = sum(lrs) / len(lrs)
                update_ratio = total_norm / total_param_norm

                loop.set_postfix_str(
                    f"loss: {accum_loss:.6f}  "
                    f"avg: {total_loss / counter:.6f}  "
                    f"avg1k: {last_thousand_sum / len(last_thousand_loss):.6f}  "
                    f"max1k: {max(last_thousand_loss):.6f}  "
                    f"step: {counter}  "
                    f"norm: {total_norm:.4f}  "
                    f"clip: {clip_norm:.4f}  "
                    f"save: {datetime.now() - last_save}  "
                    f"errors: {skipped + rolled_back}  "
                    f"lr_min: {min_lr:.2e}  "
                    f"lr_max: {max_lr:.2e}  "
                    f"lr_avg: {avg_lr:.2e}  "
                    f"upd_ratio: {update_ratio:.2e}"
                )
            accum_loss = 0.0

            if rank == 0 and datetime.now() - last_save > timedelta(hours=1):
                save(
                    model, num, optimizer, scheduler,
                    train_loss=(last_thousand_sum / len(last_thousand_loss)),
                    progress=(loop.format_dict["n"] / loop.format_dict["total"])
                )

    return total_loss / counter


def evaluate(model, loader, criterion, device):
    model.eval()
    pad_id = criterion.ignore_index
    fwd_loss_sum = 0.0
    bwd_loss_sum = 0.0
    counter = 0
    loop = tqdm(loader)
    with torch.no_grad():
        for batch in loop:
            src = batch["input"].to(device).squeeze()
            tgt = batch["output"].to(device).squeeze()
            src_mask = src.ne(pad_id)
            tgt_mask = tgt.ne(pad_id)

            output_fwd = model(src, tgt[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask[:, :-1])
            loss_fwd = criterion(output_fwd.contiguous().view(-1, vocab_size),
                                 tgt[:, 1:].contiguous().view(-1))

            output_bwd = model(tgt, src[:, :-1], src_mask=tgt_mask, tgt_mask=src_mask[:, :-1])
            loss_bwd = criterion(output_bwd.contiguous().view(-1, vocab_size),
                                 src[:, 1:].contiguous().view(-1))

            fwd_loss_sum += loss_fwd.item()
            bwd_loss_sum += loss_bwd.item()
            counter += 1

            loop.set_postfix_str(
                f"fwd: {fwd_loss_sum / counter:.6f}  "
                f"bwd: {bwd_loss_sum / counter:.6f}  "
                f"avg: {(fwd_loss_sum + bwd_loss_sum) / counter:.6f}"
            )

    return (fwd_loss_sum + bwd_loss_sum) / counter / 2, fwd_loss_sum / counter, bwd_loss_sum / counter


if __name__ == "__main__":
    PAD_ID = 3
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")
    if "LOCAL_RANK" not in os.environ:
        import subprocess
        ret = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run",
             "--nproc_per_node=2", os.path.abspath(__file__)],
            check=False,
        )
        sys.exit(ret.returncode)

    torch.set_float32_matmul_precision('high')
    train_config = json.load(open("train_config.json", mode="r"))


    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    device = f"cuda:{local_rank}"
    device_type = "cuda"

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
    batch_size = train_config["batch_size"][rank] if rank < len(train_config["batch_size"]) else train_config["batch_size"][-1]
    is_continue = train_config["continue"]
    checkpoint_dir = train_config["save_dir"]
    checkpoint_name = train_config["checkpoint"]

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
    dist.barrier()

    if rank == 0:
        print(os.path.join(checkpoint_dir, checkpoint_name))

    if is_continue:
        ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
        if not checkpoint_name:
            if rank == 0:
                print("Указано пустое имя файла при продолжении обучения! Запуск с нуля.")
            is_continue = False
        elif not os.path.isfile(ckpt_path):
            if rank == 0:
                print(f"Файл контрольной точки не найден: {ckpt_path}. Запуск с нуля.")
            is_continue = False
        else:
            if rank == 0:
                print(f"Чекпоинт: {ckpt_path}")

    if not is_continue:
        for key in config:
            if key in train_config:
                config[key] = train_config[key]
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

        param_groups = get_transformer_lrd(transformer, base_lr=train_config["init_lr"], decay=train_config["lr_decay"], weight_decay=0.02)
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-9, fused=True)
        scheduler = get_transformer_scheduler(optimizer, warmup_steps=6000)
        transformer.apply(init_weights)
        start_epoch = 0
        progress = 0
        vocab_size = config["vocab_size"]

    else:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
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

        param_groups = get_transformer_lrd(transformer, base_lr=1e-5, decay=0.8, weight_decay=0.05)
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-9, fused=True)
        scheduler = get_transformer_scheduler(optimizer, warmup_steps=6000)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        transformer.load_state_dict(ckpt["model_state_dict"], strict=False)
        if rank == 0:
            print("Контрольная точка успешно загружена!")
        progress = ckpt["progress"]
        train_loss = ckpt["train_loss"]
        start_epoch = ckpt["epoch"] - 1
        vocab_size = config["vocab_size"]

    if rank == 0:
        print("model initialized!")
        print(config)
        print(sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9)
        print(torch.cuda.memory_allocated() / 1024 ** 3)

    transformer = DDP(transformer, device_ids=[local_rank])

    _local_bs = torch.tensor(float(batch_size), device=device)
    dist.all_reduce(_local_bs, op=dist.ReduceOp.SUM)
    _total_bs = _local_bs.item()
    grad_scale = batch_size * world_size / _total_bs
    if rank == 0:
        print(f"batch_size per rank: {batch_size}, total: {int(_total_bs)}, grad_scale: {grad_scale:.2f}")

    try:
        train_dataset = load_from_disk(train_config["train_ds_dir"])
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=10, pin_memory=True, drop_last=True, persistent_workers=True,
        )

        if rank == 0:
            val_dataset = load_from_disk(train_config["val_ds_dir"])
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size,
                num_workers=6, pin_memory=True, persistent_workers=True,
            )
        else:
            val_loader = None

    except Exception as e:
        print(f"[rank {rank}] Ошибка загрузки датасета: {e}")
        dist.destroy_process_group()
        sys.exit(1)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.07)

    cold_save_counter = 0
    time.sleep(0.5)
    last_save = datetime.now()

    for epoch in range(start_epoch + 1, num_epochs + 1):
        train_sampler.set_epoch(epoch)

        if progress < 0.9:
            train_loss = train_epoch(
                transformer, train_loader, optimizer, scheduler, criterion, device, epoch,
                accumulation_steps=12
            )
        progress = 0

        if rank == 0:
            val_sum_loss, val_fwd_loss, val_bwd_loss = evaluate(transformer.module, val_loader, criterion, device)
            print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss} | Val Loss: sum-{val_sum_loss:.4f}, fwd-{val_fwd_loss:.4f}, bwd-{val_bwd_loss:.4f}")
            print(torch.cuda.memory_allocated() / 1024 ** 3)
            save(transformer, epoch, optimizer, scheduler, train_loss, val_sum_loss)

        dist.barrier()

    dist.destroy_process_group()