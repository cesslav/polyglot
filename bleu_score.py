# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")
import time
import torch
from datasets import load_dataset
from imp import Transformer
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt


CHECKPOINT_PATH = "./t5s/transformer_epoch_1.pt"
TOKENIZER_PATH = "./tokenizer/"
SRC_LANG = "rus_Cyrl"
TGT_LANG = "eng_Latn"
MAX_SRC_LEN = 512
BEAM_SIZE = 8
PAD_ID = 3
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


COMPARE_MODE = True
OWN_MODEL_NAME = "Polyglot"
HF_MODELS = [
    {"id": "Helsinki-NLP/opus-mt-ru-en"},
    {"id": "facebook/nllb-200-distilled-600M", "generate_kwargs": {"forced_bos_token_id": "eng_Latn"}},
    {"id": "facebook/nllb-200-3.3B", "generate_kwargs": {"forced_bos_token_id": "eng_Latn"}},
]
HF_BEAM_SIZE = 8
HF_MAX_LEN = 512
PLOT_ENABLED = True


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


def load_hf_model(model_id, device):
    print(f"Загрузка модели с HF: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()
    print(f"Модель {model_id} загружена: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M параметров")
    return tokenizer, model


def resolve_generate_kwargs(tokenizer, generate_kwargs):
    kwargs = dict(generate_kwargs)
    forced = kwargs.get("forced_bos_token_id")
    if isinstance(forced, str):
        if hasattr(tokenizer, "lang_code_to_id"):
            kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id[forced]
        else:
            kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(forced)
    return kwargs


def beam_search(model, tokenizer, src, src_mask, beam_size, max_len, device):
    bos, eos = 0, 1
    with torch.no_grad():
        memory = model.encoder(src, mask=src_mask)
        beams = [(torch.tensor([[bos]], device=device), 0.0)]

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == eos:
                    new_beams.append((seq, score))
                    continue
                logits = model.to_logits(model.decoder(seq, memory, context_mask=src_mask))
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


def make_own_translate_fn(model, tokenizer, beam_size, max_len, device):
    def translate_one(text):
        src = tokenizer(
            text,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = 512,
        )["input_ids"].to(device)

        src_mask = src.ne(PAD_ID)
        return beam_search(model, tokenizer, src, src_mask, beam_size, max_len, device)

    return translate_one


def make_hf_translate_fn(model, tokenizer, beam_size, max_len, device, generate_kwargs=None):
    generate_kwargs = generate_kwargs or {}

    def translate_one(text):
        inputs = tokenizer(
            text,
            return_tensors = "pt",
            truncation = True,
            max_length = max_len,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length = max_len,
                num_beams = beam_size,
                **generate_kwargs,
            )
        return tokenizer.decode(generated[0], skip_special_tokens=True)

    return translate_one


def evaluate(translate_fn, src_sentences, ref_sentences, desc="Перевод"):
    hypotheses = []
    t_start = time.perf_counter()
    total_chars = 0

    for src_text in tqdm(src_sentences, desc=desc):
        hypothesis = translate_fn(src_text)
        hypotheses.append(hypothesis)
        total_chars += len(hypothesis)

    elapsed = time.perf_counter() - t_start

    bleu = BLEU()
    bleu_sentence = BLEU(effective_order=True)
    corpus_result = bleu.corpus_score(hypotheses, [ref_sentences])
    sentence_bleu = [bleu_sentence.sentence_score(h, [r]).score for h, r in zip(hypotheses, ref_sentences)]

    return corpus_result, hypotheses, sentence_bleu, elapsed, total_chars


def run_comparison(models, src_sentences, ref_sentences, split):
    results = {}
    for name, translate_fn in models:
        print(f"\nПеревод корпуса моделью «{name}» ({split})...")
        corpus_result, hypotheses, sentence_bleu, elapsed, total_chars = evaluate(
            translate_fn, src_sentences, ref_sentences, desc=f"{name} [{split}]"
        )
        results[name] = {
            "corpus": corpus_result,
            "hypotheses": hypotheses,
            "sentence_bleu": sentence_bleu,
            "elapsed": elapsed,
            "total_chars": total_chars,
        }
    return results


def format_comparison_table(results, model_names):
    headers = ["Модель", "BLEU", "min", "avg", "max"]
    rows = []
    for name in model_names:
        sentence_bleu = results[name]["sentence_bleu"]
        rows.append([
            name,
            f"{results[name]['corpus'].score:.2f}",
            f"{min(sentence_bleu):.2f}",
            f"{sum(sentence_bleu) / len(sentence_bleu):.2f}",
            f"{max(sentence_bleu):.2f}",
        ])

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def format_row(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [format_row(headers), "-+-".join("-" * w for w in widths)]
    lines += [format_row(row) for row in rows]
    return lines


def plot_comparison(results, model_names, save_path):
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib не установлен, график пропущен.")
        return

    plt.figure(figsize=(12, 6))
    for name in model_names:
        plt.plot(results[name]["sentence_bleu"], label=name, linewidth=1, alpha=0.8)
    plt.xlabel("Номер примера")
    plt.ylabel("BLEU")
    plt.title("BLEU каждой модели по примерам")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"График сохранён: {save_path}")
    try:
        plt.show()
    except Exception as e:
        print(f"Не удалось отобразить график (нет доступного дисплея?): {e}")
    plt.close()


def write_compare_log(log_path, own_config, split, src_sentences, ref_sentences, results, model_names):
    n = len(src_sentences)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint (Polyglot): {CHECKPOINT_PATH}\n")
        f.write(f"Config (Polyglot):     d_model={own_config['d_model']} layers={own_config['num_layers']} heads={own_config['num_heads']}\n")
        f.write(f"Split:                 FLORES+ {split}\n")
        f.write(f"Sentences:             {n}\n")
        f.write(f"Модели:                {', '.join(model_names)}\n\n")

        for name in model_names:
            corpus_result = results[name]["corpus"]
            elapsed = results[name]["elapsed"]
            f.write(f"[{name}] {corpus_result}\n")
            f.write(f"[{name}] Время: {elapsed:.1f} с | {n / elapsed:.1f} пр/с\n\n")

        for line in format_comparison_table(results, model_names):
            f.write(f"{line}\n")

        f.write("\n--- Примеры переводов (сравнение моделей) ---\n")
        f.write("--- Формат: [исходный текст, перевод модели 1, ..., перевод модели N, ожидаемый перевод] ---\n\n")
        for i in range(min(20, n)):
            row = [src_sentences[i]] + [results[name]["hypotheses"][i] for name in model_names] + [ref_sentences[i]]
            f.write(f"[{i + 1}] {row}\n")

    print(f"Результаты сравнения сохранены в {log_path}")


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

    own_translate_fn = make_own_translate_fn(model, tokenizer, BEAM_SIZE, MAX_SRC_LEN, DEVICE)

    hf_models = []
    if COMPARE_MODE:
        if not HF_MODELS:
            print("COMPARE_MODE включён, но HF_MODELS пуст — будет оценена только модель Polyglot.")
        for hf_cfg in HF_MODELS:
            model_id = hf_cfg["id"]
            hf_tokenizer, hf_model = load_hf_model(model_id, DEVICE)
            generate_kwargs = resolve_generate_kwargs(hf_tokenizer, hf_cfg.get("generate_kwargs", {}))
            hf_models.append((
                model_id,
                make_hf_translate_fn(hf_model, hf_tokenizer, HF_BEAM_SIZE, HF_MAX_LEN, DEVICE, generate_kwargs),
            ))

    for split in ("dev", "devtest"):
        src_sentences, ref_sentences = load_flores(SRC_LANG, TGT_LANG, split=split)
        n = len(src_sentences)

        if not COMPARE_MODE:
            corpus_result, hypotheses, sentence_bleu, elapsed, total_chars = evaluate(
                own_translate_fn, src_sentences, ref_sentences, desc="Перевод"
            )

            print(f"\n{'─' * 52}")
            print(f"  Сплит:           FLORES+ {split}")
            print(f"  Предложений:     {n}")
            print(f"  BLEU:            {corpus_result.score:.2f}")
            print(f"  {corpus_result}")
            print(f"  Время:           {elapsed:.1f} с")
            print(f"  Скорость:        {n / elapsed:.1f} пр/с | {total_chars / elapsed:.0f} chr/s")
            print(f"{'=' * 52}\n")

            log_path = f"bleu_{split}.txt"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
                f.write(f"Config:     d_model={config['d_model']} layers={config['num_layers']} heads={config['num_heads']}\n")
                f.write(f"Split:      FLORES+ {split}\n")
                f.write(f"Sentences:  {n}\n")
                f.write(f"BLEU:       {corpus_result.score:.2f}\n")
                f.write(f"{corpus_result}\n")
                f.write(f"Time:       {elapsed:.1f} s\n")
                f.write(f"Speed:      {n / elapsed:.1f} sent/s | {total_chars / elapsed:.0f} chr/s\n")
                f.write("\n--- Примеры переводов ---\n\n")
                for i in range(min(20, n)):
                    f.write(f"[{i+1}] SRC: {src_sentences[i]}\n")
                    f.write(f"     HYP: {hypotheses[i]}\n")
                    f.write(f"     REF: {ref_sentences[i]}\n\n")

            print(f"Результаты сохранены в {log_path}")

        else:
            models = [(OWN_MODEL_NAME, own_translate_fn)] + hf_models
            model_names = [name for name, _ in models]

            results = run_comparison(models, src_sentences, ref_sentences, split)

            print(f"\n{'─' * 52}")
            print(f"  Сплит:           FLORES+ {split}")
            print(f"  Предложений:     {n}\n")
            for line in format_comparison_table(results, model_names):
                print(f"  {line}")
            print(f"\n{'=' * 52}\n")

            log_path = f"bleu_compare_{split}.txt"
            write_compare_log(log_path, config, split, src_sentences, ref_sentences, results, model_names)

            if PLOT_ENABLED:
                plot_comparison(results, model_names, f"bleu_compare_{split}.png")
