# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")
import time
import torch
from datasets import load_dataset
from imp import Transformer
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoTokenizer


CHECKPOINT_PATH = "./t5s/transformer_epoch_1.pt"
TOKENIZER_PATH = "./tokenizer/"
SRC_LANG = "rus_Cyrl"
TGT_LANG = "eng_Latn"
MAX_SRC_LEN = 256
BEAM_SIZE = 10
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = checkpoint["config"]

    model = Transformer(
        dim = config["d_model"],
        enc_num_tokens = config["vocab_size"],
        enc_depth = config["num_layers"],
        enc_heads = config["num_heads"],
        enc_dim_head = config["dim_head"],
        enc_mlp_mult = config["mlp_mult"],
        dec_num_tokens = config["vocab_size"],
        dec_depth = config["num_layers"] + config["dec_depth_diff"],
        dec_heads = config["num_heads"],
        dec_dim_head = config["dim_head"],
        dec_mlp_mult = config["mlp_mult"],
        dropout = config["dropout"],
        tie_token_emb = True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def beam_search(model, tokenizer, src, beam_size, max_len, device):
    bos, eos = 0, 1
    with torch.no_grad():
        memory = model.encoder(src)
        beams = [(torch.tensor([[bos]], device=device), 0.0)]

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == eos:
                    new_beams.append((seq, score))
                    continue
                logits = model.to_logits(model.decoder(seq, memory))
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                topk_lp, topk_t = torch.topk(log_probs, beam_size, dim=-1)
                for k in range(beam_size):
                    next_tok = topk_t[0, k].unsqueeze(0).unsqueeze(0)
                    new_beams.append((
                        torch.cat([seq, next_tok], dim=1),
                        score + topk_lp[0, k].item(),
                    ))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(seq[0, -1].item() == eos for seq, _ in beams):
                break

    return tokenizer.decode(beams[0][0][0], skip_special_tokens=True)


def evaluate(model, tokenizer, src_sentences, ref_sentences, beam_size, max_len, device):
    hypotheses = []
    references = [ref_sentences]
    t_start = time.perf_counter()
    total_chars = 0

    for src_text in tqdm(src_sentences, desc="Перевод"):
        src = tokenizer(
            src_text,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = 512,
        )["input_ids"].to(device)

        hypothesis = beam_search(model, tokenizer, src, beam_size, max_len, device)
        hypotheses.append(hypothesis)
        total_chars += len(hypothesis)

    elapsed = time.perf_counter() - t_start
    bleu = BLEU()
    result = bleu.corpus_score(hypotheses, references)

    return result, hypotheses, elapsed, total_chars


def load_flores(src_lang, tgt_lang, split="devtest"):
    print(f"Загрузка FLORES+ ({split}, {src_lang} / {tgt_lang})...")
    ds_src = load_dataset("openlanguagedata/flores_plus", src_lang, split=split)
    ds_tgt = load_dataset("openlanguagedata/flores_plus", tgt_lang, split=split)
    src_sentences = [ex["text"] for ex in ds_src]
    ref_sentences = [ex["text"] for ex in ds_tgt]
    print(f"Загружено {len(src_sentences)} пар предложений.")
    return src_sentences, ref_sentences


if __name__ == "__main__":
    print(f"Устройство: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model, config = load_model(CHECKPOINT_PATH, DEVICE)
    print(f"Модель загружена: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M параметров")
    print(f"Конфигурация: d_model={config['d_model']}, layers={config['num_layers']}, heads={config['num_heads']}")
    print()

    for split in ("dev", "devtest"):
        src_sentences, ref_sentences = load_flores(SRC_LANG, TGT_LANG, split=split)

        bleu_result, hypotheses, elapsed, total_chars = evaluate(
            model, tokenizer, src_sentences, ref_sentences,
            beam_size = BEAM_SIZE,
            max_len = MAX_SRC_LEN,
            device = DEVICE,
        )

        n = len(src_sentences)
        print(f"\n{'─' * 52}")
        print(f"  Сплит:           FLORES+ {split}")
        print(f"  Предложений:     {n}")
        print(f"  BLEU:            {bleu_result.score:.2f}")
        print(f"  {bleu_result}")
        print(f"  Время:           {elapsed:.1f} с")
        print(f"  Скорость:        {n / elapsed:.1f} пр/с | {total_chars / elapsed:.0f} chr/s")
        print(f"{'=' * 52}\n")

        log_path = f"bleu_{split}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
            f.write(f"Config:     d_model={config['d_model']} layers={config['num_layers']} heads={config['num_heads']}\n")
            f.write(f"Split:      FLORES+ {split}\n")
            f.write(f"Sentences:  {n}\n")
            f.write(f"BLEU:       {bleu_result.score:.2f}\n")
            f.write(f"{bleu_result}\n")
            f.write(f"Time:       {elapsed:.1f} s\n")
            f.write(f"Speed:      {n / elapsed:.1f} sent/s | {total_chars / elapsed:.0f} chr/s\n")
            f.write("\n--- Примеры переводов ---\n\n")
            for i in range(min(20, n)):
                f.write(f"[{i+1}] SRC: {src_sentences[i]}\n")
                f.write(f"     HYP: {hypotheses[i]}\n")
                f.write(f"     REF: {ref_sentences[i]}\n\n")

        print(f"Результаты сохранены в {log_path}")