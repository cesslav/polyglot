import os
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.client.session.aiohttp import AiohttpSession
from tqdm import tqdm
from transformers import AutoTokenizer
from v8_imp import Transformer

TOKEN = "7991454522:AAFm4tWsTglDNBGcfYPTCqcyFx-zdZktqig"

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

# ---------- загрузка токенизатора ----------
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/mixed48k")

# ---------- загрузка модели ----------
checkpoint_dir = "t5s"
checkpoint_name = "transformer_epoch_1.pt"

checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), weights_only=False, map_location=device)
config = checkpoint["config"]
# transformer = Transformer(config["src_vocab_size"], config["tgt_vocab_size"], config["d_model"], config["num_heads"], config["num_layers"], config["d_ff"], config["max_seq_length"], 0.2).to(device)
model = Transformer(
    dim=config["d_model"],
    # max_seq_len = 1024,
    enc_num_tokens=config["vocab_size"],
    enc_depth=config["num_layers"],
    enc_heads=config["num_heads"],
    enc_dim_head=config["dim_head"],
    enc_mlp_mult=config["mlp_mult"],
    dec_num_tokens=config["vocab_size"],
    dec_depth=config["num_layers"],
    dec_heads=config["num_heads"],
    dec_dim_head=config["dim_head"],
    dec_mlp_mult=config["mlp_mult"],
    dropout=config["dropout"],
    tie_token_emb=True
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

bos = 0
eos = 1
pad = 3

# ---------- функция генерации ----------

def generate(text, max_len=128):

    src = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")["input_ids"].to(device)
    print(text)

    with torch.no_grad():
        tgt = torch.tensor([[bos]], device=device)
        enc = model.encoder(src)
        for _ in range(max_len):
            dec = model.decoder(tgt, enc)
            logits = model.to_logits(dec)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if next_token.item() == eos:
                break
    a = tokenizer.decode(tgt[0], skip_special_tokens=True)
    print(a)
    return a  #


# ---------- aiogram ----------
# session = AiohttpSession(proxy="http://s5.wyckoff.one:443")

bot = Bot(TOKEN)   # , session=session
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Отправь текст, и я переведу его.")


@dp.message()
async def translate(message: Message):

    text = message.text

    try:
        translation = generate(text)

        await message.answer(translation)

    except Exception as e:
        await message.answer(f"Ошибка перевода: {e}")


# ---------- запуск ----------

if __name__ == "__main__":
    import asyncio
    print("Started.")
    asyncio.run(dp.start_polling(bot))