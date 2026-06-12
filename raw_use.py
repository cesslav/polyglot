# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")

import os
from transformers import AutoTokenizer
import torch
from imp import Transformer


def beam_search(transformer, tokenizer, src, src_mask, beam_size=5, max_len=256,
                repetition_penalty=1.3, device="cpu"):
    bos, eos = 0, 1
    with torch.no_grad():
        enc = transformer.encoder(src, mask=src_mask)
        beams = [(torch.tensor([[bos]], device=device), 0.0)]

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == eos:
                    new_beams.append((seq, score))
                    continue

                dec = transformer.decoder(seq, enc, context_mask=src_mask)
                next_token_logits = transformer.to_logits(dec)[0, -1, :].clone()

                for token_id in set(seq[0].tolist()):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty

                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

                for k in range(beam_size):
                    new_seq = torch.cat([seq, topk_tokens[k].view(1, 1)], dim=1)
                    new_beams.append((new_seq, score + topk_log_probs[k].item()))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            if all(b[0][0, -1].item() == eos for b in beams):
                break

        best_seq = max(beams, key=lambda x: x[1] / len(x[0][0]))[0]

    return tokenizer.decode(best_seq[0], skip_special_tokens=True)


tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")

checkpoint_dir = "./t5s"
checkpoint_name = "transformer_epoch_1.pt"
device = "cpu"

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
).to(device)

transformer.load_state_dict(checkpoint["model_state_dict"])
transformer.eval()

pad = 3
beam_size = 1

while True:
    print("Input:")
    text = input()
    src = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=512, return_tensors="pt"
    )["input_ids"].to(device)
    src_mask = src.ne(pad)

    output = beam_search(
        transformer, tokenizer, src, src_mask,
        beam_size=beam_size, device=device
    )
    print("Output:")
    print(output)