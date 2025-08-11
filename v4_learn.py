from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from v4_imp import Transformer, tokenize_fn
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler
torch.set_float32_matmul_precision('high')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC_LANG = "ru"
TGT_LANG = "en"
N_HEADS = 8
MODEL_DIM = N_HEADS * 32
NUM_LAYERS = 24
FF_DIM = MODEL_DIM * 4
DROPOUT = 0.1
BATCH_SIZE = 6
MAX_LEN = 512
NUM_EPOCHS = 50
LR = 1e-4
save_coef = 0.025


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", cache_dir="./sources/tokenizers")

dataset = load_dataset("wmt/wmt19", f"{SRC_LANG}-{TGT_LANG}", split="train")
dataset.set_format(type='torch')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)


model = Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=MODEL_DIM,
    n_heads=N_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT
).to(DEVICE)
model.compile()
# model = torch.compile(model, fullgraph=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
print(torch.cuda.memory_allocated() / 1024 ** 3)
ds_len = len(dataloader)
print(ds_len, ds_len * save_coef)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    iterations = 0
    saves = 0
    loop = tqdm(dataloader, mininterval=1)

    for batch in loop:
        iterations += 1
        pack = tokenize_fn(batch, SRC_LANG, TGT_LANG, MAX_LEN, tokenizer)
        input_ids = pack['input_ids'].to(DEVICE)
        labels = pack['labels'].to(DEVICE)

        tgt_input = labels[:, :-1]
        tgt_output = labels[:, 1:]

        logits = model(input_ids, tgt_input, tokenizer.pad_token_id, DEVICE)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        loop.set_postfix(loss=loss.item(), saves=saves, epoch=epoch, refresh=False)

        if iterations >= ds_len * save_coef:
            iterations = 0
            saves += 1
            avg_loss = total_loss / (ds_len * (save_coef * saves))
            torch.save(model.state_dict(), f"./content/{epoch}_{saves}")


    avg_loss = total_loss / ds_len
    print(f"model saved at {str(datetime.now())[11:-7]} after epoch N{epoch} with avg loss {avg_loss:.4f}")
    torch.save(model.state_dict(), f"./content/{epoch}")
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")