import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def tokenize_fn(example, src_lang, tgt_lang, max_len, tokenizer):
    src_text = example["translation"][src_lang]
    tgt_text = example["translation"][tgt_lang]
    src = tokenizer(src_text, truncation=True, padding='max_length', max_length=max_len, return_tensors="pt")
    tgt = tokenizer(tgt_text, truncation=True, padding='max_length', max_length=max_len, return_tensors="pt")
    return {
        'input_ids': src['input_ids'].squeeze(),
        'attention_mask': src['attention_mask'].squeeze(),
        'labels': tgt['input_ids'].squeeze()
    }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)  # 💡 РЕШЕНИЕ

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key_value=None, mask=None):
        if key_value is None:
            key_value = query  # self-attention

        B, L_q, _ = query.size()
        B, L_kv, _ = key_value.size()

        # Применение линейных проекций
        q = self.q_proj(query).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, L_q, d_k)
        k = self.k_proj(key_value).view(B, L_kv, self.num_heads, self.d_k).transpose(1, 2)  # (B, h, L_kv, d_k)
        v = self.v_proj(key_value).view(B, L_kv, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, L_q, L_kv)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, L_q, L_kv)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)  # (B, h, L_q, L_kv)
        context = torch.matmul(attn, v)  # (B, h, L_q, d_k)

        # 🛠 ПРИМЕНИТЬ ПРАВИЛЬНЫЙ RESHAPE
        context = context.transpose(1, 2).contiguous()  # (B, L_q, h, d_k)
        context = context.view(B, L_q, self.num_heads * self.d_k)  # (B, L_q, d_model)

        return self.out_proj(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.self_attn(x, mask=mask)
        x = x + self.dropout(x2)
        x = self.norm1(x)
        x2 = self.ff(x)
        x = x + self.dropout(x2)
        x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask):
        x2 = self.self_attn(x, mask=tgt_mask)
        x = x + self.dropout(x2)
        x = self.norm1(x)

        x2 = self.cross_attn(x, key_value=memory, mask=memory_mask)
        x = x + self.dropout(x2)
        x = self.norm2(x)

        x2 = self.ff(x)
        x = x + self.dropout(x2)
        x = self.norm3(x)
        return x

def generate_subsequent_mask(size, device):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask.to(device)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, ff_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        return self.fc_out(tgt)