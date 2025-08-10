from transformers import AutoTokenizer, M2M100ForConditionalGeneration

def translate(text, weights="./checkpoint_ru_en_1.2B", device="cuda"):
    model = M2M100ForConditionalGeneration.from_pretrained(weights).to(device)
    tokenizer = AutoTokenizer.from_pretrained(weights)
    model_inputs = tokenizer(text, return_tensors="pt").to("cuda")
    gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("en"))
    return tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    iterations = 1
    texts_to_translate = ["привет! как дела?", "Что делаешь на выходных?", "Как себя чувствует Олег?", "Я выиграл в лотерее!"]

    for j in range(iterations):
        print(f"iteration #{j}")
        for i in texts_to_translate:
            print(translate(i))