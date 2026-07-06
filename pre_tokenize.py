# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch
import numpy as np
from tqdm import tqdm

tensor_size = 512
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
SRC_LANG = "rus_Cyrl"
TGT_LANG = "eng_Latn"
PAD_ID = tokenizer.pad_token_id or 3

FILTER_CONFIG = {
    "length": {
        "enabled": True,
        "min_tokens": 2,
        "max_tokens": 512,
    },
    "length_ratio": {
        "enabled": True,
        "max_ratio": 2,
    },
    "langid": {
        "enabled": False,
        "src_lang": "ru",
        "tgt_lang": "en",
        "model_path": "",
    },
    "labse": {
        "enabled": True,
        "min_cosine": 0.9,
        "model_name": "sentence-transformers/LaBSE",
        "batch_size": 512,
    }
}

_lid_model = None
_labse_model = None
_labse_pool = None
_cometkiwi_model = None


def _get_lid_model():
    global _lid_model
    if _lid_model is None:
        import fasttext
        fasttext.FastText.eprint = lambda x: None
        _lid_model = fasttext.load_model(FILTER_CONFIG["langid"]["model_path"])
    return _lid_model


def _get_labse_model():
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer(FILTER_CONFIG["labse"]["model_name"])
    return _labse_model


def _get_labse_pool():
    global _labse_pool
    if _labse_pool is None:
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)][:2] if num_gpus > 0 else ["cpu"]
        _labse_pool = _get_labse_model().start_multi_process_pool(target_devices=devices)
    return _labse_pool


def _stage1_length_ratio(example):
    cfg = FILTER_CONFIG["length"]
    if cfg["enabled"]:
        if len(example["input"]) > cfg["max_tokens"] or len(example["output"]) > cfg["max_tokens"]:
            return False
        if _non_pad_count(example["input"]) < cfg["min_tokens"]:
            return False
        if _non_pad_count(example["output"]) < cfg["min_tokens"]:
            return False
    if FILTER_CONFIG["length_ratio"]["enabled"]:
        src_len = len(example["src_text"])
        tgt_len = len(example["tgt_text"])
        if min(src_len, tgt_len) == 0:
            return False
        if max(src_len, tgt_len) / min(src_len, tgt_len) > FILTER_CONFIG["length_ratio"]["max_ratio"]:
            return False
    return True


def _stage2_langid(example):
    model = _get_lid_model()
    src_clean = example["src_text"].replace("\n", " ").strip()
    tgt_clean = example["tgt_text"].replace("\n", " ").strip()
    if not src_clean or not tgt_clean:
        return False
    src_pred = model.predict(src_clean, k=1)[0][0].replace("__label__", "")
    tgt_pred = model.predict(tgt_clean, k=1)[0][0].replace("__label__", "")
    return src_pred == FILTER_CONFIG["langid"]["src_lang"] and tgt_pred == FILTER_CONFIG["langid"]["tgt_lang"]


def _stage3_labse(batch):
    model = _get_labse_model()
    src_emb = model.encode(batch["src_text"], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    tgt_emb = model.encode(batch["tgt_text"], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    scores = (src_emb * tgt_emb).sum(axis=1)
    return (scores >= FILTER_CONFIG["labse"]["min_cosine"]).tolist()


def _run_labse_bulk(dataset):
    model = _get_labse_model()
    pool = _get_labse_pool()
    cfg = FILTER_CONFIG["labse"]
    chunk = cfg.get("chunk_size", cfg["batch_size"] * 64)
    src_texts = dataset["src_text"]
    tgt_texts = dataset["tgt_text"]
    n = len(src_texts)
    keep = []
    bar = tqdm(total=n, desc="LaBSE")
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        src_emb = model.encode(src_texts[start:end], pool=pool, batch_size=cfg["batch_size"],
                               normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        tgt_emb = model.encode(tgt_texts[start:end], pool=pool, batch_size=cfg["batch_size"],
                               normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        scores = (np.array(src_emb) * np.array(tgt_emb)).sum(axis=1)
        keep.extend((np.where(scores >= cfg["min_cosine"])[0] + start).tolist())
        bar.update(end - start)
    bar.close()
    return dataset.select(keep)


def tokenization_wmt(example, num):
    src_text = example["translation"][SRC_LANG[0:2]]
    tgt_text = example["translation"][TGT_LANG[0:2]]
    return {
        "input": np.array(tokenizer(src_text, padding="max_length", max_length=tensor_size)["input_ids"],
                          dtype=np.uint16),
        "output": np.array(tokenizer(tgt_text, padding="max_length", max_length=tensor_size)["input_ids"],
                           dtype=np.uint16),
        "src_text": src_text,
        "tgt_text": tgt_text,
    }


def tokenization_fine(example, num):
    src_text = example["og_full_text"]
    tgt_text = example["translated_text"]
    coef = 1 if example["og_quality_score"] > 0.5 else 0
    if coef:
        return {
            "input": np.array(tokenizer(src_text, padding="max_length", max_length=tensor_size)["input_ids"], dtype=np.uint16),
            "output": np.array(tokenizer(tgt_text, padding="max_length", max_length=tensor_size)["input_ids"], dtype=np.uint16),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    else:

        pad = np.full(tensor_size+1, PAD_ID, dtype=np.uint16)
        return {"input": pad, "output": pad, "src_text": "", "tgt_text": ""}



def tokenization_flores(example, num):
    src_text = ds[pairs[0]][num]["text"]
    tgt_text = ds[pairs[1]][num]["text"]
    return {
        "input": np.array(tokenizer(src_text, padding="max_length", max_length=tensor_size)["input_ids"],
                          dtype=np.uint16),
        "output": np.array(tokenizer(tgt_text, padding="max_length", max_length=tensor_size)["input_ids"],
                           dtype=np.uint16),
        "src_text": src_text,
        "tgt_text": tgt_text,
    }


def tokenization_tatoeba(example, num):
    lang_src = example["lang_src"]
    lang_tgt = example["lang_tgt"]
    if lang_src == SRC_LANG[0:3] and lang_tgt == TGT_LANG[0:3]:
        src_text, tgt_text = example["sentence_src"], example["sentence_tgt"]
    elif lang_src == TGT_LANG[0:3] and lang_tgt == SRC_LANG[0:3]:
        src_text, tgt_text = example["sentence_tgt"], example["sentence_src"]
    else:
        pad = np.full(tensor_size + 1, PAD_ID, dtype=np.uint16)
        return {"input": pad, "output": pad, "src_text": "", "tgt_text": ""}
    return {
        "input": np.array(tokenizer(src_text, padding="max_length", max_length=tensor_size)["input_ids"],
                          dtype=np.uint16),
        "output": np.array(tokenizer(tgt_text, padding="max_length", max_length=tensor_size)["input_ids"],
                           dtype=np.uint16),
        "src_text": src_text,
        "tgt_text": tgt_text,
    }


def _non_pad_count(ids):
    if isinstance(ids, list):
        return len(ids) - ids.count(PAD_ID)
    return int((np.asarray(ids) != PAD_ID).sum())


def apply_quality_filters(dataset, num_proc_cheap=20, skip_model_filters=False):
    print(f"Входной размер: {len(dataset)}")
    if FILTER_CONFIG["length"]["enabled"]:
        dataset = dataset.filter(_stage1_length_ratio, num_proc=num_proc_cheap)
        print(f"После этапа 1 (длина + ratio): {len(dataset)}")
    if FILTER_CONFIG["langid"]["enabled"]:
        dataset = dataset.filter(_stage2_langid, num_proc=num_proc_cheap)
        print(f"После этапа 2 (langid): {len(dataset)}")
    if FILTER_CONFIG["labse"]["enabled"]:
        dataset = _run_labse_bulk(dataset)
        print(f"После этапа 3 (LaBSE): {len(dataset)}")
    dataset = dataset.remove_columns(["src_text", "tgt_text"])
    dataset.set_format(type="torch", columns=["input", "output"])
    print(f"Итоговый размер: {len(dataset)}")
    return dataset

if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

    ds_wmt = load_dataset("wmt/wmt19", f"{SRC_LANG[0:2]}-{TGT_LANG[0:2]}", split="train")
    wmt_dataset = ds_wmt.map(tokenization_wmt, with_indices=True, num_proc=20)
    wmt_dataset = apply_quality_filters(wmt_dataset)

    ds_fine = load_dataset("HuggingFaceFW/finetranslations", f"{SRC_LANG}", split="train").remove_columns(["id", "og_chunks", "translated_chunks", "og_language", "og_language_score", "og_token_count", "early_stop", "url", "warc_path", "minhash_cluster_size", "translated_token_count", "edu_score_raw", "edu_score"])
    print(ds_fine)
    fine_dataset = ds_fine.map(tokenization_fine, with_indices=True, num_proc=20)
    fine_dataset = apply_quality_filters(fine_dataset)

    ds_short = load_dataset("ymoslem/Tatoeba-Translations", split="train")
    short_dataset = ds_short.map(tokenization_tatoeba, with_indices=True, num_proc=20)
    short_dataset = apply_quality_filters(short_dataset)

    train_dataset = concatenate_datasets([fine_dataset, wmt_dataset, short_dataset])
    train_dataset.save_to_disk("./sources/s512_clear")


    ds = [
        load_dataset("openlanguagedata/flores_plus", f"{SRC_LANG}", split="dev"),
        load_dataset("openlanguagedata/flores_plus", f"{TGT_LANG}", split="dev"),
        load_dataset("openlanguagedata/flores_plus", f"{SRC_LANG}", split="devtest"),
        load_dataset("openlanguagedata/flores_plus", f"{TGT_LANG}", split="devtest"),
    ]

    pairs = [0, 1]
    dataset1 = load_dataset("openlanguagedata/flores_plus", f"{SRC_LANG}", split="dev").map(tokenization_flores, with_indices=True)
    dataset1 = apply_quality_filters(dataset1, num_proc_cheap=10)

    pairs = [2, 3]
    dataset2 = load_dataset("openlanguagedata/flores_plus", f"{SRC_LANG}", split="devtest").map(tokenization_flores, with_indices=True)
    dataset2 = apply_quality_filters(dataset2, num_proc_cheap=10)

    eval_dataset = concatenate_datasets([dataset1, dataset2])
    eval_dataset.save_to_disk("./sources/s512_val")


    if _labse_pool is not None:
        _get_labse_model().stop_multi_process_pool(_labse_pool)