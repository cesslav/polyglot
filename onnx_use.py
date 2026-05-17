# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import sys
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def np_softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class ONNXTransformer:
    def __init__(self, encoder_path, decoder_path):
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)

    def encode(self, src):
        return self.encoder.run(["memory"], {"src": src.astype(np.int64)})[0]

    def decode(self, tgt, memory):
        return self.decoder.run(
            ["logits"],
            {"tgt": tgt.astype(np.int64), "memory": memory.astype(np.float32)},
        )[0]


def beam_search_stream(model, tokenizer, src, beam_size=4, max_len=128):
    np.log_softmax = lambda x, axis: np.log(np_softmax(x))
    bos, eos = 0, 1

    memory = model.encode(src.cpu().numpy())
    beams = [(np.array([[bos]], dtype=np.int64), 0.0)]

    num_out_tokens = 0
    t_start = time.perf_counter()

    for _ in range(max_len):
        new_beams = []

        for seq, score in beams:
            if seq[0, -1] == eos:
                new_beams.append((seq, score))
                continue

            logits = model.decode(seq, memory)
            log_probs = np.log_softmax(logits[:, -1, :], axis=-1)
            topk_idx = np.argsort(-log_probs, axis=-1)[0][:beam_size]
            topk_log_probs = log_probs[0][topk_idx]

            for k in range(beam_size):
                new_seq = np.concatenate([seq, [[topk_idx[k]]]], axis=1)
                new_score = score + float(topk_log_probs[k])
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        current_text = tokenizer.decode(beams[0][0][0], skip_special_tokens=True)

        num_out_tokens += 1
        yield current_text

        if all(seq[0, -1] == eos for seq, _ in beams):
            break

    elapsed = time.perf_counter() - t_start
    full_text = tokenizer.decode(beams[0][0][0], skip_special_tokens=True)
    num_chars = len(full_text)

    return full_text, num_out_tokens, num_chars, elapsed


def print_stats(num_in_tokens, padding, num_out_tokens, num_chars, elapsed):
    tok_per_sec = num_out_tokens / elapsed if elapsed > 0 else 0
    char_per_sec = num_chars / elapsed if elapsed > 0 else 0
    print(
        f"\n\033[90m"
        f"[{elapsed:.2f} с | "
        f"{tok_per_sec:.1f} tok/s | "
        f"{char_per_sec:.1f} chr/s | "
        f"вход: {num_in_tokens} tok (pad {padding}) | "
        f"выход: {num_out_tokens} tok, {num_chars} chr]"
        f"\033[0m"
    )


if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
    model = ONNXTransformer(
        encoder_path = "./encoder.onnx",
        decoder_path = "./decoder.onnx",
    )

    print("Модель загружена. Введите текст для перевода (Ctrl+C для выхода).")

    while True:
        print("Input:")
        text = input()
        if not text.strip():
            continue

        raw_tokens = tokenizer(text, return_tensors="pt")["input_ids"]
        num_in_tokens = raw_tokens.shape[1]
        max_length = next((n for n in range(64, 513, 64) if n >= num_in_tokens), 512)

        src = tokenizer(
            text,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = max_length,
        )["input_ids"]

        print("\nOutput:")

        gen = beam_search_stream(model, tokenizer, src, beam_size=1)
        prev_len = 0
        full_text = ""
        num_out_tokens = 0
        num_chars = 0
        elapsed = 0.0

        try:
            while True:
                current_text = next(gen)

                sys.stdout.write("\r" + " " * prev_len + "\r")
                sys.stdout.write(current_text)
                sys.stdout.flush()

                prev_len = len(current_text)

        except StopIteration as e:
            full_text, num_out_tokens, num_chars, elapsed = e.value

        sys.stdout.write("\r" + " " * prev_len + "\r")
        sys.stdout.write(full_text)

        print_stats(num_in_tokens, max_length, num_out_tokens, num_chars, elapsed)
        print()