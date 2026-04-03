import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from torch.utils.data import DataLoader
from v8_imp import Transformer
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

os.environ['HIP_VISIBLE_DEVICES'] = "0"
os.environ['HSA_OVERRIDE_GFX_VERSION'] = "11.0.0"


def get_transformer_scheduler(optimizer, warmup_steps):

    def lr_lambda(step):

        step = max(step, 1)

        if step < warmup_steps:
            return step / warmup_steps

        return (warmup_steps ** 0.5) / (step ** 0.5)

    return LambdaLR(optimizer, lr_lambda)


def get_transformer_lrd(model, base_lr=1e-4, decay=0.9):
    encoder_layers = len(model.encoder.layer)
    decoder_layers = len(model.decoder.layer)
    max_depth = encoder_layers + decoder_layers + 1
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

        elif "to_logits" in name:
            depth = max_depth

        else:
            depth = max_depth

        if depth not in layer_map:
            layer_map[depth] = []

        layer_map[depth].append(param)
    param_groups = []
    for depth, params in layer_map.items():
        lr = base_lr * (decay ** (max_depth - depth))
        param_groups.append({
            "params": params,
            "lr": lr
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


def save(transformer, epoch, optimizer, scheduler, train_loss=0, val_loss=0, progress=0):
    global last_save, config
    # print(datetime.now())
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

    last_save = datetime.now()

    torch.save(checkpoint, os.path.join(checkpoint_dir, f"transformer_epoch_{epoch}.pt"))


def train_epoch(model, loader, optimizer, scheduler, criterion, device, num):
    model.train()
    total_loss = 0
    last_thousand_loss = []
    counter = 0
    loop = tqdm(loader)
    for batch in loop:
        optimizer.zero_grad()
        src, tgt = batch["input"].to(device).squeeze(), batch["output"].to(device).squeeze()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            output = model(src, tgt[:, :-1]) / 0.7# , src_mask)
            loss = criterion(output.contiguous().view(-1, vocab_size), tgt[:, 1:].contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        counter += 1

        if len(last_thousand_loss) < 1000:
            last_thousand_loss.append(loss.item())
        else:
            last_thousand_loss.pop(0)
            last_thousand_loss.append(loss.item())
        loop.set_postfix_str(f"loss: {loss.item():.6f}, avg_loss: {(total_loss / counter):.6f}  "
                             f"{(sum(last_thousand_loss) / len(last_thousand_loss)):.6f}  "
                             f"{max(last_thousand_loss):.6f}, since_last_save: {datetime.now() - last_save}")
        if datetime.now() - last_save > timedelta(hours=2):
            save(model, num, optimizer, scheduler, total_loss / counter, progress=(loop.format_dict["n"] / loop.format_dict["total"]))
    return total_loss / len(loader)


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


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
num_epochs = 200
base_lr = 2e-4            # lr для top (последних) слоёв / head
lr_decay = 0.85            # множитель снижения lr для каждого шага глубины

checkpoint_dir = "./t5m" # "t5xs"
checkpoint_name = ""
os.makedirs(checkpoint_dir, exist_ok=True)
criterion = nn.CrossEntropyLoss(ignore_index=3, label_smoothing=0.1)



if checkpoint_name:
    checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), weights_only=False)
    config = checkpoint["config"]
    # transformer = Transformer(config["src_vocab_size"], config["tgt_vocab_size"], config["d_model"], config["num_heads"], config["num_layers"], config["d_ff"], config["max_seq_length"], 0.2).to(device)
    transformer = Transformer(
        dim=config["d_model"],
        # max_seq_len = 1024,
        enc_num_tokens=config["vocab_size"],
        enc_depth=config["num_layers"],
        enc_heads=config["num_heads"],
        enc_dim_head=config["dim_head"],
        enc_mlp_mult=config["mlp_mult"],
        dec_num_tokens=config["vocab_size"],
        dec_depth=config["num_layers"]+2,
        dec_heads=config["num_heads"],
        dec_dim_head=config["dim_head"],
        dec_mlp_mult=config["mlp_mult"],
        dropout=config["dropout"],
        tie_token_emb=True
    ).to(device)

    param_groups = get_transformer_lrd(transformer, base_lr=base_lr, decay=lr_decay)
    optimizer = optim.Adam(param_groups, betas=(0.9, 0.98), eps=1e-9)  # , lr=1e-4
    scheduler = get_transformer_scheduler(optimizer, warmup_steps=12000)
    # torch.autograd.set_detect_anomaly(True)
    # optimizer = optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    transformer.load_state_dict(checkpoint["model_state_dict"])
    progress = checkpoint["progress"]
    train_loss = checkpoint["train_loss"]
    start_epoch = checkpoint["epoch"]
    vocab_size = config["vocab_size"]

else:
    dim = 896
    vocab_size = 48000
    layers = 6
    dim_head = 96
    heads = int(dim / dim_head)
    mlp_mult = 10
    dropout = 0.15
    progress = 0

    config = {
        "vocab_size": vocab_size,
        "d_model": dim,
        "mlp_mult": mlp_mult,
        "num_heads": heads,
        "dim_head": dim_head,
        "num_layers": layers,
        "dropout": dropout
    }

    transformer = Transformer(
        dim=dim,
        enc_num_tokens=vocab_size,
        enc_depth=layers,
        enc_heads=heads,
        enc_dim_head=dim_head,
        enc_mlp_mult=mlp_mult,
        dec_num_tokens=vocab_size,
        dec_depth=layers+2,
        dec_heads=heads,
        dec_dim_head=dim_head,
        dec_mlp_mult=mlp_mult,
        dropout=dropout,
        tie_token_emb=True
    ).to(device)


    param_groups = get_transformer_lrd(transformer, base_lr=base_lr, decay=lr_decay)
    optimizer = optim.Adam(param_groups, betas=(0.9, 0.98), eps=1e-9)  # transformer.parameters(), lr=1e-4
    scheduler = get_transformer_scheduler(optimizer, warmup_steps=30000)
    # torch.autograd.set_detect_anomaly(True)
    transformer.apply(init_weights)
    start_epoch = 0


print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000000)
print(torch.cuda.memory_allocated() / 1024**3)

time.sleep(0.5)
last_save = datetime.now()

# train_dataset = load_from_disk("/home/trashdata/sources/s1024_full")
# test_dataset = load_from_disk("/home/trashdata/sources/s1024_val")

train_loader = DataLoader(load_from_disk("./sources/s1024_full"), batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
# train_loader = DataLoader(load_from_disk("/home/trashdata/sources/s1024_val"), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(load_from_disk("./sources/s1024_val"), batch_size=batch_size, num_workers=12, pin_memory=True)


for epoch in range(start_epoch+1, num_epochs + 1):
    if progress < 0.9:
        train_loss = train_epoch(transformer, train_loader, optimizer, scheduler, criterion, device, epoch)
    progress = 0
    val_loss = evaluate(transformer, val_loader, criterion, device)

    print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(torch.cuda.memory_allocated() / 1024 ** 3)

    save(transformer, epoch, optimizer, train_loss, val_loss)

#  0%|          | 7999/4748169 [49:07<483:59:26,  2.72it/s, loss: 6.638919, avg_loss: 7.881477  6.805798  8.069850, since_last_save: 0:54:44.756761]