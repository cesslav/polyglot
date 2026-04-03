import torch
from transformers import AutoTokenizer
from tokenizers import normalizers
# ============================
# Конфигурация
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("../tokenizer/polyglot_tokenizer", cache_dir="./sources/tokenizers")

# model = Transformer(
#     n_src_vocab=16192,
#     src_pad_idx=1
# ).to(device)

model = torch.load("./checkpoints_1024/0_13.pt", weights_only=False).to(device)

print(torch.cuda.memory_allocated() / 1024**3)

def translate(text, model, tokenizer, device):
    src_tok = tokenizer(text, truncation=True, padding='max_length', max_length=model.max_seq_len, return_tensors="pt")['input_ids'].to(device)
    bos = torch.tensor([1]).to(device)
    out_tok = model(src_tok, bos)
    # print(out_tok)


    # print(out_tok)
    # print(len(out_tok[0]))
    return tokenizer.decode(out_tok[0], skip_special_tokens=True)

def translate_qq(text, model, tokenizer, device):
    norm = normalizers.Sequence([normalizers.NFD()])
    src_tok = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors="pt")['input_ids'].tolist()[0]
    out = tokenizer.decode(src_tok, skip_special_tokens=True)
    # out = out.replace(text, "")
    return out

phrases = ["Cъешь же ещё этих мягких французских булок, да выпей чаю!",
           "Всё ускоряющаяся эволюция компьютерных технологий предъявила жёсткие требования к производителям как собственно вычислительной техники, так и периферийных устройств.",
           "Привет, как дела?",
           "Вступив в бой с шипящими змеями — эфой и гадюкой, — маленький, цепкий, храбрый ёж съел их.",
           "БУКВОПЕЧАТАЮЩЕЙ СВЯЗИ НУЖНЫ ХОРОШИЕ Э/МАГНИТНЫЕ РЕЛЕ. ДАТЬ ЦИФРЫ (1234567890+= .?-)",
           "Cozy sphinx waves quart jug of bad milk.",
           "The jay, pig, fox, zebra and my wolves quack!",
           "Crazy Fredrick bought many very exquisite opal jewels?"]


if __name__ == "__main__":
    for text in phrases:
        print(translate(text, model, tokenizer, device))