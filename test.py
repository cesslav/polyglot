from transformers import pipeline

pipe = pipeline("text-generation", model="qwen/qwen3.5-0.8B")
messages = [
    {"role": "user", "content": "Напиши доклад по теме биология 11 класс"
                                ""},
]
a = pipe(messages)

print(a)