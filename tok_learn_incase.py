# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from transformers import PreTrainedTokenizerFast


ds = load_dataset("wmt/wmt19", f"ru-en", split="train", cache_dir="/home/trashdata/HF/cache", streaming=True)
batch = 100
print(ds)


def apply_inline_casing(text):
    alpha_all = [c for c in text if c.isalpha()]
    if len(alpha_all) > 1 and all(c.isupper() for c in alpha_all):
        return "<|upall|> " + text.lower()

    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isupper():
            j = i
            while j < n and text[j].isupper():
                j += 1
            run = text[i:j]
            out.append("<|up|>" if len(run) > 1 else "<|cap|>")
            out.append(run.lower())
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def restore_inline_casing(text):
    if text.startswith("<|upall|> "):
        return text[len("<|upall|> "):].upper()

    out = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("<|cap|>", i):
            i += len("<|cap|>")
            if i < n:
                out.append(text[i].upper())
                i += 1
            continue
        if text.startswith("<|up|>", i):
            i += len("<|up|>")
            while i < n and text[i].isalpha():
                out.append(text[i].upper())
                i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def get_training_corpus(batch=10):
    for i in ds.iter(batch_size=batch):
        yield [apply_inline_casing(t) for t in i["og_full_text"]]
        yield [apply_inline_casing(t) for t in i["translated_text"]]


tokenizer = Tokenizer(models.Unigram())
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's mixed48k pre-tokenization!"))
print(apply_inline_casing("Привет, Мир! Это ТЕСТ."))
print(apply_inline_casing("ВНИМАНИЕ, ПРОВЕРКА СВЯЗИ!"))
print(apply_inline_casing("Смотрю YouTube и покупаю на eBay."))

special_tokens = ["<|bos|>", "<|eos|>", "<|unk|>", "<|pad|>", "<|mask|>", "<|s|>", "<|/s|>", "<|up|>", "<|cap|>", "<|upall|>"]
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

test_text = apply_inline_casing("Раз, два, три. Проверка связи. Как слышно?")
encoding = tokenizer.encode(test_text)
print(encoding.tokens)
print(encoding.ids)
print(tokenizer.decode(encoding.ids))
print(restore_inline_casing(tokenizer.decode(encoding.ids)))

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    unk_token="<|unk|>",
    pad_token="<|pad|>",
    cls_token="<|cls|>",
    sep_token="<|sep|>",
    mask_token="<|mask|>",
    padding_side="right",
)

wrapped_tokenizer.save_pretrained("./tokenizer/")
