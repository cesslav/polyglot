import warnings
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
# import torch.autograd.profiler as profiler
# from transformers import AutoTokenizer
from tqdm import tqdm
from legacy.v5_imp import Transformer

torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


debug = False
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 5
SRC_LANG = "ru"
TGT_LANG = "en"
save_coef = 0.005

if debug:
    device = "cpu"
    BATCH_SIZE = 5
    LR = 1

b = torch.tensor([[1] * 256] * BATCH_SIZE).to(device)
# b = [bos]
# for i in range(1, BATCH_SIZE):
#     b.append(bos)
print(b.size())


dataset = load_from_disk("../sources/wmt19")
# dataset.set_format(type='torch')
# ru_tokenizer = AutoTokenizer.from_pretrained("./ru_tokenizer/polyglot_tokenizer")
# norm = normalizers.Sequence([normalizers.NFD()])
# mixed48k = ru_tokenizer("Привет, мир!", truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].to(device)
# print(mixed48k)

model = Transformer(
    n_src_vocab=9960,
    batch_size=BATCH_SIZE
).to(device)
model.compile()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
# print(dataloader[0]["input"])


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss(reduction="sum")
# print(ru_tokenizer.is_fast)
print(f"model (predicted) size: {(torch.cuda.memory_allocated() / 1024 ** 3):.3f} GB.")
ds_len = len(dataloader)
print("Examples in dataset:", str(ds_len) + ".")
print("Model will saved every", int(ds_len * save_coef), "examples.")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    iterations = 0
    saves = 0
    loop = tqdm(dataloader)

    for batch in loop:
        # with profiler.profile(use_device = 'cuda') as prof:
        iterations += 1
        # input_ids = ru_tokenizer(batch["translation"][SRC_LANG], truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].squeeze().to(device)
        # target_ids = ru_tokenizer(batch["translation"][TGT_LANG], truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"].squeeze().to(device)
        input_ids = batch["input"].to(device)
        # print(input_ids.size())
        target_ids = batch["output"].to(device)
        # b = [bos * BATCH_SIZE]
        # print(b)
        # print(input_ids.size())
        # print(target_ids.size())
        # print(bos.size())

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids, b).squeeze().to(device)
        # print(logits)
        # print(logits.size())
        # print(target_ids.size())

        loss = criterion(logits.float(), target_ids.float())
        loss.requires_grad = True

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item(), saves=saves, epoch=epoch, refresh=False)

        if iterations >= ds_len * save_coef:
            iterations = 0
            saves += 1
            avg_loss = total_loss / (ds_len * (save_coef * saves))
            torch.save(model, f"./checkpoints/{epoch}_{saves}.pt")
            try:
                pass
                # print(ru_tokenizer.decode(model(mixed48k, bos)))
            except Exception as e:
                print(e)

        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        # print("_______________________________________________________")


    avg_loss = total_loss / ds_len
    print(f"model saved at {str(datetime.now())[11:-7]} after epoch N{epoch} with avg loss {avg_loss:.4f}")
    torch.save(model, f"./checkpoints/{epoch}.pt")