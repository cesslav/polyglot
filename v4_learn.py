from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from v4_imp import Transformer, tokenize_fn, generate_subsequent_mask

# 3. Обучение
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SRC_LANG = "ru"
    TGT_LANG = "en"
    N_HEADS = 8
    MODEL_DIM = N_HEADS * 32
    NUM_LAYERS = 16
    FF_DIM = MODEL_DIM * 4
    DROPOUT = 0.15
    BATCH_SIZE = 8
    MAX_LEN = 512
    NUM_EPOCHS = 50
    LR = 1e-4


    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", cache_dir="./sources/tokenizers")

    dataset = load_dataset("wmt/wmt19", f"{SRC_LANG}-{TGT_LANG}", split="train[:1%]")
    dataset.set_format(type='torch')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)


    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=MODEL_DIM,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print(torch.cuda.memory_allocated() / 1024 ** 3)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader):
            pack = tokenize_fn(batch, SRC_LANG, TGT_LANG, MAX_LEN)
            input_ids = pack['input_ids'].to(DEVICE)
            labels = pack['labels'].to(DEVICE)

            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]

            src_mask = (input_ids != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
            tgt_mask = generate_subsequent_mask(tgt_input.size(1), DEVICE)

            logits = model(input_ids, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        torch.save(model.state_dict(), f"./content/{str(datetime.now())[11:-7]}_{avg_loss:.4f}")
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

