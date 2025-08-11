import torch
from v4_imp import Transformer, tokenize_fn  # твоя реализация
from transformers import AutoTokenizer

# ============================
# Конфигурация
# ============================
print(torch.cuda.is_available())
device = torch.device("cpu") #  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры модели
N_HEADS = 8
MODEL_DIM = N_HEADS * 32
NUM_LAYERS = 24
FF_DIM = MODEL_DIM * 4
DROPOUT = 0.1
BATCH_SIZE = 6
MAX_LEN = 512

# Путь к модели
MODEL_PATH = "./content/0_1_6.9811"

# Используем Huggingface токенизатор
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", cache_dir="./sources/tokenizers")


bos_id = tokenizer.bos_token_id
if bos_id is None:
    bos_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.pad_token_id

eos_id = tokenizer.eos_token_id
if eos_id is None:
    eos_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.pad_token_id

pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0


# ============================
# Загрузка обученной модели
# ============================

model = Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=MODEL_DIM,
    n_heads=N_HEADS,
    num_layers=NUM_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT
)


model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)



"""def translate(text, device, tokenizer):
    src_tok = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors="pt")['input_ids'].squeeze()
    pad = tokenizer("<|endoftext|>", return_tensors="pt")['input_ids']
    output = model(src_tok, pad, tokenizer.pad_token_id, device)
    out_text = tokenizer.decode(output, skip_special_tokens=True)
    return out_text"""


def translate(text, max_len=MAX_LEN):
    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len
        )

        src_ids = src_tokens["input_ids"].to(device)

        tgt_ids = torch.tensor([[bos_id]], device=device)

        print("Максимальный токен в src:", src_ids.max().item())
        print("Максимальный токен в tgt:", tgt_ids.max().item())
        print("Размер словаря модели:", model.embedding.num_embeddings)

        for _ in range(max_len):
            output = model(src_ids, tgt_ids, pad_token=pad_id, device=device)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

            if next_token.item() == eos_id:
                break

        translation = tokenizer.decode(tgt_ids.squeeze(), skip_special_tokens=True)
        return translation


if __name__ == "__main__":
    src_text = "Hello, how are you?"
    translation = translate(src_text)
    print(f"Input: {src_text}")
    print(f"Translation: {translation}")