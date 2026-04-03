# from v6_imp import Transformer
from v61_imp import Seq2Seq
from os import listdir
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("../tokenizer/polyglot_tokenizer")
# point = listdir("./proto/")[0]
# model = Transformer.load_from_checkpoint(f"./proto/{point}",
#                  n_src_vocab=9960,
#                  src_pad_idx=0,
#                  d_model=768,
#                  d_inner=3072,
#                  n_layers=24,
#                  n_head=12,
#                  d_k=96,
#                  d_v=96,
#                  dropout=0.1,
#                  n_position=256,
#                  d_word_vec=None,
#                  tgt_pad_idx=None,
#                  n_tgt_vocab=None,
#                  tgt_emb_prj_weight_sharing=True,
#                  emb_src_tgt_weight_sharing=True,
#                  scale_emb_or_prj='emb')

model = Seq2Seq() # .to("cuda")
torch.set_printoptions(profile="full")

to_trans = "Мама мыла раму."
tokens = tokenizer(to_trans, truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"] #  .to("cuda")
zeros = tokenizer("", truncation=True, padding='max_length', max_length=256, return_tensors="pt")["input_ids"] #  .to("cuda")
out_tokens = model(tokens, zeros).squeeze()
print(out_tokens.size())
sampled = torch.distributions.multinomial.Multinomial(probs=out_tokens).sample().max(-1).indices.tolist()
for i in sampled:
    output = tokenizer.decode(i)
    print(output, end="")