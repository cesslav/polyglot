from transformers import AutoTokenizer
from datasets import load_dataset,concatenate_datasets, load_from_disk, Dataset
import torch

tensor_size = 512
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/mixed48k")


def tokenization_fine(example, num):
    return {
        "input": tokenizer(example['translation']["ru"], return_tensors="pt", padding="max_length", max_length=tensor_size)["input_ids"],
        "output": tokenizer(example['translation']["en"], return_tensors="pt", padding="max_length", max_length=tensor_size)["input_ids"]
    }


def tokenization_flores(example, num):
    return {
        "input": tokenizer(ds[pairs[0]][num]["text"], return_tensors="pt", padding="max_length", max_length=tensor_size)["input_ids"],
        "output": tokenizer(ds[pairs[1]][num]["text"], return_tensors="pt", padding="max_length", max_length=tensor_size)["input_ids"]
    }


def tokenization_tatoeba(example, num):
    lang_src = example["lang_src"]
    lang_tgt = example["lang_tgt"]
    if lang_src == "rus" and lang_tgt == "eng":
        return {
            "input": tokenizer(example["sentence_src"], return_tensors="pt", padding="max_length", max_length=tensor_size)["input_ids"],
            "output": tokenizer(example["sentence_tgt"], return_tensors="pt", padding="max_length", max_length=tensor_size)["input_ids"]}
    elif lang_src == "eng" and lang_tgt == "rus":
        return {
            "input": tokenizer(example["sentence_tgt"], return_tensors="pt", padding="max_length", max_length=tensor_size)["input_ids"],
            "output": tokenizer(example["sentence_src"], return_tensors="pt", padding="max_length", max_length=tensor_size)["input_ids"]}
    else:
        return {
            "input": torch.tensor([[3]*(tensor_size*2)]),
            "output": torch.tensor([[3]*(tensor_size*2)])
        }


def length_filter(example):
    return example["input"].size(1) == tensor_size and example["output"].size(1) == tensor_size


ds_fine = load_dataset("wmt/wmt19", "ru-en", split="train", cache_dir="/home/trashdata/HF/cache")  # , cache_dir="/home/trashdata/HF/cache"
print(ds_fine)
main_dataset = ds_fine.map(tokenization_fine, num_proc=20, with_indices=True)
main_dataset.set_format(type="torch", columns=["input", "output"])
main_dataset = main_dataset.filter(length_filter, num_proc=20)


ds_short = load_dataset("ymoslem/Tatoeba-Translations", split="train", cache_dir="/home/trashdata/HF/cache")   # .select(range(100000))  # .take(24000000)
short_dataset = ds_short.map(tokenization_tatoeba, num_proc=16, with_indices=True)
short_dataset.set_format(type="torch", columns=["input", "output"])
short_dataset = short_dataset.filter(length_filter, num_proc=16)


train_dataset = concatenate_datasets([main_dataset, short_dataset])
train_dataset.save_to_disk("/home/trashdata/sources/s1024_full")



SRC_LANG = "rus_Cyrl"
TGT_LANG = "eng_Latn"

ds = [load_dataset("openlanguagedata/flores_plus", f"{SRC_LANG}", split="dev", cache_dir="/home/trashdata/HF/cache"),
      load_dataset("openlanguagedata/flores_plus", f"{TGT_LANG}", split="dev", cache_dir="/home/trashdata/HF/cache"),
      load_dataset("openlanguagedata/flores_plus", f"{SRC_LANG}", split="devtest", cache_dir="/home/trashdata/HF/cache"),
      load_dataset("openlanguagedata/flores_plus", f"{TGT_LANG}", split="devtest", cache_dir="/home/trashdata/HF/cache")
      ]

pairs = [0, 1]
dataset1 = load_dataset("openlanguagedata/flores_plus", f"{SRC_LANG}", split="dev", cache_dir="/home/trashdata/HF/cache").map(tokenization_flores, num_proc=16, with_indices=True)
dataset1.set_format(type="torch", columns=["input", "output"])
pairs = [2, 3]
dataset2 = load_dataset("openlanguagedata/flores_plus", f"{SRC_LANG}", split="devtest", cache_dir="/home/trashdata/HF/cache").map(tokenization_flores, num_proc=16, with_indices=True)
dataset2.set_format(type="torch", columns=["input", "output"])

eval_dataset = concatenate_datasets([dataset1, dataset2])
eval_dataset.save_to_disk("./sources/s1024_val")


eval_dataset.save_to_disk("/home/trashdata/sources/s1024_short_full")
