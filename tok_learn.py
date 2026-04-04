# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")


from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors


ds = load_dataset("wmt/wmt19", f"ru-en", split="train", cache_dir="/home/trashdata/HF/cache", streaming=True)
batch = 100
print(ds)
def get_training_corpus(batch=10):
    for i in ds.iter(batch_size=batch):
        yield i["og_full_text"]
        yield i["translated_text"]


tokenizer = Tokenizer(models.Unigram())
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

wrapped_tokenizer.save_pretrained("./tokenizer/")