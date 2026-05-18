# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import torch
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from transformers import AutoTokenizer
import numpy as np
import onnxruntime as ort


TOKEN = input("Введите токен API телеграма для своего бота: ")
bot = Bot(TOKEN)
dp = Dispatcher()


class ONNXTransformer:
    def __init__(self, encoder_path, decoder_path, device="cpu"):
        providers = ["CPUExecutionProvider"]

        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

    def encode(self, src):
        inputs = {
            "src": src.astype(np.int64),
        }

        memory = self.encoder.run(["memory"], inputs)[0]
        return memory

    def decode(self, tgt, memory):
        inputs = {
            "tgt": tgt.astype(np.int64),
            "memory": memory.astype(np.float32),
        }

        logits = self.decoder.run(["logits"], inputs)[0]
        return logits


def beam_search_onnx(model, tokenizer, src, beam_size=4, max_len=256):
    np.log_softmax = lambda x, axis: np.log(np_softmax(x))
    bos, eos = 0, 1
    src_np = src.cpu().numpy()
    memory = model.encode(src_np)
    beams = [(np.array([[bos]], dtype=np.int64), 0.0)]
    for _ in range(max_len):
        new_beams = []

        for seq, score in beams:
            if seq[0, -1] == eos:
                new_beams.append((seq, score))
                continue

            logits = model.decode(seq, memory)

            next_token_logits = logits[:, -1, :]
            log_probs = np.log_softmax(next_token_logits, axis=-1)

            topk_idx = np.argsort(-log_probs, axis=-1)[0][:beam_size]
            topk_log_probs = log_probs[0][topk_idx]

            for k in range(beam_size):
                next_token = topk_idx[k]
                new_seq = np.concatenate([seq, [[next_token]]], axis=1)
                new_score = score + float(topk_log_probs[k])

                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(seq[0, -1] == eos for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return tokenizer.decode(best_seq[0], skip_special_tokens=True)


def np_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Отправь текст, и я переведу его.")


@dp.message()
async def translate(message: Message):
    text = message.text

    try:
        src = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")[
            "input_ids"].to(device)
        translation = beam_search_onnx(model, tokenizer, src, beam_size=4)

        await message.answer(translation)

    except Exception as e:
        await message.answer(f"Ошибка перевода: {e}")



if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/mixed48k")
    model = ONNXTransformer(
        encoder_path="onnx_export/encoder_int8.onnx",
        decoder_path="onnx_export/decoder_int8.onnx"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import asyncio
    print("Started.")
    asyncio.run(dp.start_polling(bot))