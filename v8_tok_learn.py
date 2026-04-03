import sys
import os
from numpy.ma.extras import column_stack
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tokenizers import normalizers, Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from tqdm import tqdm


"""vocab_size = 32000
batch_size = 1

os.environ["TRANSFORMERS_CACHE"] = "/home/trashdata/HF/cache"
dataset = load_dataset("HuggingFaceFW/finetranslations", f"rus_Cyrl", split=["train"], columns=["translated_text", "og_full_text"], cache_dir="/home/trashdata/HF/cache")[0].select(range(200000))
# dataset2 = load_dataset("Helsinki-NLP/news_commentary", f"en-ru", split=["train"])[0]["text"]
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3n-E4B-it")
print(dataset[0])
print("source vocabulary size", tokenizer.vocab_size)
print("target vocabulary size", vocab_size)
# print(ru_tokenizer.is_fast)
# print("num examples in dataset", len(dataloader))

def batch_iterator():
    for text in dataset:
        yield text["translated_text"]
        yield text["og_full_text"]


new_ru_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=vocab_size, with_indices=True)
print("_________")
new_ru_tokenizer.save_pretrained("./tokenizer/polyglot_tokenizer")"""

# dataset1 = load_dataset("Helsinki-NLP/news_commentary", f"rus_Cyrl", split=["dev"])[0]["text"]
# dataset2 = load_dataset("Helsinki-NLP/news_commentary", f"rus_Cyrl", split=["devtest"])[0]["text"]



# norm = normalizers.Sequence([normalizers.NFD()])
# print(new_ru_tokenizer(norm.normalize_str(dataset["ru"])))
# print(new_en_tokenizer(norm.normalize_str(dataset["en"])))

ds = load_dataset("/home/ceslav/.cache/huggingface/hub/datasets--HuggingFaceFW--finetranslations", f"default", columns=["translated_text", "og_full_text"], cache_dir="/home/trashdata/HF/cache", streaming=True)["train"]  #  , split=["train"]
batch = 1000
print(ds)
# sys.exit(0)
def get_training_corpus(batch=10):
    for i in ds.iter(batch_size=batch):  #  range(0, len(ds))
        # print(ds[i:i + batch])
        yield i["og_full_text"]
        yield i["translated_text"]

tokenizer = Tokenizer(models.Unigram())
# tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's mixed48k pre-tokenization!"))

special_tokens = ["<|bos|>", "<|eos|>", "<|unk|>", "<|pad|>", "<|mask|>", "<|s|>", "<|/s|>"]
trainer = trainers.UnigramTrainer(vocab_size=48000, special_tokens=special_tokens, unk_token="<|unk|>", show_progress=True)
tokenizer.train_from_iterator(get_training_corpus(batch), trainer=trainer)
bos_token_id = tokenizer.token_to_id("<|bos|>")
eos_token_id = tokenizer.token_to_id("<|eos|>")
print(bos_token_id, eos_token_id)
tokenizer.post_processor = processors.TemplateProcessing(
    single="<|bos|>:0 $A:0 <|eos|>:0",
    pair="<|bos|>:0 $A:0 <|eos|>:0 <|bos|>:1 $B:1 <|eos|>:1",
    special_tokens=[("<|bos|>", bos_token_id), ("<|eos|>", eos_token_id)],
)
tokenizer.decoder = decoders.Metaspace()

encoding = tokenizer.encode("Раз, два, три. Проверка связи. Как слышно?")
print(encoding.tokens)
print(encoding.ids)
print(tokenizer.decode(encoding.ids))

from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|s|>",
    eos_token="<|/s|>",
    unk_token="<|unk|>",
    pad_token="<|pad|>",
    cls_token="<|cls|>",
    sep_token="<|sep|>",
    mask_token="<|mask|>",
    padding_side="right",
)

wrapped_tokenizer.save_pretrained("./tokenizer/mixed48k")

wrapped_tokenizer = AutoTokenizer.from_pretrained("tokenizer/mixed48k")
print(wrapped_tokenizer(["Раз, два, три. Проверка связи.", "Как слышно? Приём!"], truncation=True, padding='max_length', max_length=25))