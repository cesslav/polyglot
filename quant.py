import torch
from transformers import AutoTokenizer, M2M100ForConditionalGeneration


weights="./checkpoint_ru_en_1.2B"
model = M2M100ForConditionalGeneration.from_pretrained(weights)
tokenizer = AutoTokenizer.from_pretrained(weights)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

for i in range(100):
    model_inputs = tokenizer(input("/"), return_tensors="pt")
    gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("en"))
    print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))