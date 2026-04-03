import torch
import torch.nn as nn
import torch.optim as optim
from v7_imp_mod import Transformer


src_vocab_size = 9960
tgt_vocab_size = 9960
d_model = 1024
d_ff = d_model * 4
num_heads = 8
num_layers = 12
max_seq_length = 256
dropout = 0.2
batch_size = 8
device = "cuda"

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
print((next(transformer.parameters()).device))

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(device)  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(device)  # (batch_size, seq_length)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(200):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


transformer.eval()

# Generate random sample validation data
val_src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(device)  # (batch_size, seq_length)
val_tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(device)  # (batch_size, seq_length)

with torch.no_grad():

    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")