# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from transformers import PreTrainedTokenizerFast


def get_training_corpus(batch=10):
    for i in dataset.iter(batch_size=batch):
        yield i["input"]
        yield i["output"]


tokenizer = Tokenizer(models.Unigram())
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
special_tokens = ["<|bos|>", "<|eos|>", "<|unk|>", "<|pad|>", "<|mask|>"]
trainer = trainers.UnigramTrainer(vocab_size=48000, special_tokens=special_tokens, unk_token="<|unk|>", show_progress=True)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
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

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    unk_token="<|unk|>",
    pad_token="<|pad|>",
    mask_token="<|mask|>",
    padding_side="right",
)

wrapped_tokenizer.save_pretrained("./tokenizer/")