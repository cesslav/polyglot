import torch
import torch.nn.functional as Fun
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from v7_imp import TransformerL
from v7_conf import TransformerConfig


def prediction(x, y):
    """This gives the probability distribution over English words for a given German translation"""
    with torch.inference_mode():
        logits = model(x, y)
    # print(logits)
    # print(logits.size())
    soft = nn.Softmax(dim=-1)
    logits = soft(logits)
    # print(logits)
    # print(logits.size())
    return logits


def translate(input_sentence):
    """This function generates the translation"""

    tokenized_input = tokenizer(input_sentence, truncation=True, padding='max_length', max_length=256,
                                return_tensors="pt")["input_ids"].type(torch.int)

    decoded_sentence = ""
    # print(tokenized_input)
    for i in range(0, config.block_size):
        tokenized_target_sentence = tokenizer(decoded_sentence, truncation=True, padding='max_length', max_length=256,
                                    return_tensors="pt")["input_ids"].type(torch.int)
        # print(tokenized_input.dtype)
        # print(tokenized_target_sentence.dtype)
        # print(tokenized_target_sentence)
        predictions = prediction(tokenized_input.to(config.device), tokenized_target_sentence.to(config.device)).squeeze()
        # print(predictions.size())
        # print(predictions)
        sampled_token_index = torch.multinomial(predictions[i, :], num_samples=1).item()
        sampled_token = tokenizer.decode(sampled_token_index)

        decoded_sentence += sampled_token
        print(sampled_token, end="")

        if sampled_token == "<eos>":
            break
    return decoded_sentence


class style:
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


tokenizer = AutoTokenizer.from_pretrained("../tokenizer/polyglot_tokenizer/")
name = "./checkpoints_1024/model_post_0"
state = torch.load(name, weights_only=False)
config = TransformerConfig()


test_dataset = load_from_disk("./sources/wmt19_s256_val").take(1500)
# test_dataset = torch.load("test_dataset.pt", weights_only=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Set up the transformer and load the past state
model = TransformerL(config).to(config.device)

# Loads the model
model.load_state_dict(state["model_state_dict"])
print("num params(B):", sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000000000)
print("model size(GB):", torch.cuda.memory_allocated() / 1024**3)

model.eval()

with torch.no_grad():
    for i, elem in enumerate(test_dataloader):
        if i % 100 == 0:
            # print(elem[0].size())
            inp = elem["input"].squeeze()
            out = elem["output"].squeeze()
            #  print(inp)
            #  print(out)
            print(style.BOLD + "Orginal" + style.END)
            german = tokenizer.decode(inp).replace("<pad>", "").replace("<bos>", "")
            print(german)
            print(style.BOLD + "Translation" + style.END)
            print(tokenizer.decode(out).replace("<pad>", "").replace("<bos>", ""))
            print(style.BOLD + "Machine Translation" + style.END)
            print(translate(german))
            print("\n")