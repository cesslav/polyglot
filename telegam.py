import os
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import Message
from transformers import AutoTokenizer
from v8_imp import Transformer

TOKEN = "Telegram Bot Token"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("./tokenizer")


checkpoint_dir = "./"
checkpoint_name = "transformer.pt"

checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), weights_only=False, map_location=device)
config = checkpoint["config"]
model = Transformer(
    dim=config["d_model"],
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
beam_size = 4


def beam_search(transformer, tokenizer, src, beam_size=10, max_len=256, device="cpu"):
    bos, eos = 0, 1
    with torch.no_grad():
        enc = transformer.encoder(src)
        # (sequence, score)
        beams = [(torch.tensor([[bos]], device=device), 0.0)]
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == eos:
                    new_beams.append((seq, score))
                    continue
                dec = transformer.decoder(seq, enc)
                logits = transformer.to_logits(dec)
                next_token_logits = logits[:, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)
                for k in range(beam_size):
                    next_token = topk_tokens[0, k].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_log_probs[0, k].item()
                    new_beams.append((new_seq, new_score))
            # сортируем по score
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            # если все завершены — стоп
            if all(seq[0, -1].item() == eos for seq, _ in beams):
                break
        best_seq = beams[0][0]
    return tokenizer.decode(best_seq[0])


bot = Bot(TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Отправь текст, и я переведу его.")


@dp.message()
async def translate(message: Message):
    text = message.text

    try:
        src = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")[
            "input_ids"].to(device)
        translation = beam_search(model, tokenizer, src, beam_size=beam_size, device=device)

        await message.answer(translation)

    except Exception as e:
        await message.answer(f"Ошибка перевода: {e}")



if __name__ == "__main__":
    import asyncio
    print("Started.")
    asyncio.run(dp.start_polling(bot))