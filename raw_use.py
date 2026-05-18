# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")


import os
from transformers import AutoTokenizer
import torch
from imp import Transformer


def beam_search(transformer, tokenizer, src, beam_size=10, max_len=256, device="cpu"):
    bos, eos = 0, 1
    with torch.no_grad():
        enc = transformer.encoder(src)
        # (sequence, score)
        beams = [(torch.tensor([[bos]], device=device), 0.0)]
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == eos:
                    new_beams.append((seq, score))
                    continue
                dec = transformer.decoder(seq, enc)
                logits = transformer.to_logits(dec)
                next_token_logits = logits[:, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)
                for k in range(beam_size):
                    next_token = topk_tokens[0, k].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + topk_log_probs[0, k].item()
                    new_beams.append((new_seq, new_score))
            # сортируем по score
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            # если все завершены — стоп
            if all(seq[0, -1].item() == eos for seq, _ in beams):
                break
        best_seq = beams[0][0]
    return tokenizer.decode(best_seq[0], skip_special_tokens=True)


tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")

checkpoint_dir = "./t5s"
checkpoint_name = "transformer_epoch_1.pt"
device =  "cpu"


checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_name), weights_only=False, map_location=device)
config = checkpoint["config"]

transformer = Transformer(
    dim=config["d_model"],
    enc_num_tokens=config["vocab_size"],
    enc_depth=config["num_layers"],
    enc_heads=config["num_heads"],
    enc_dim_head=config["dim_head"],
    enc_mlp_mult=config["mlp_mult"],
    dec_num_tokens=config["vocab_size"],
    dec_depth=config["num_layers"] + config["dec_depth_diff"],
    dec_heads=config["num_heads"],
    dec_dim_head=config["dim_head"],
    dec_mlp_mult=config["mlp_mult"],
    dropout=config["dropout"],
    tie_token_emb=True
).to(device).to(torch.float16)

transformer.load_state_dict(checkpoint["model_state_dict"])
transformer.eval()
torch.cuda.empty_cache()


bos = 0
eos = 1
pad = 3
max_len = 64
beam_size = 1


while True:
    print("Input:")
    text = input()
    src = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")["input_ids"].to(device)
    for i in range(1):
        with torch.no_grad():
            tgt = torch.tensor([[bos]], device=device)
            enc = transformer.encoder(src)
            output = beam_search(transformer, tokenizer, src, beam_size=beam_size, device=device)
            print("Output:")
            print(output)