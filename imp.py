# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")


import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


def default(val, d):
    return val if val is not None else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        # self.register_buffer("beta", torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.dim = dim

    def forward(self, x):
        # return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
        return F.layer_norm(x, (self.dim,), self.gamma, self.beta)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()

        inner_dim = int(dim * mult)

        self.w1 = nn.Linear(dim, inner_dim)
        self.w2 = nn.Linear(dim, inner_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.out = nn.Linear(inner_dim, dim)

    def forward(self, x):

        x1 = self.w1(x)
        x2 = self.w2(x)

        x = F.silu(x1) * x2

        x = self.dropout1(x)

        x = self.out(x)

        x = self.dropout2(x)

        return x


class RelativePositionBias(nn.Module):

    def __init__(self, scale, causal, num_buckets=32, max_distance=128, heads=12):
        super().__init__()

        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

        self._cache = {}

    @staticmethod
    def _relative_position_bucket(
            relative_position,
            causal=True,
            num_buckets=32,
            max_distance=128
    ):

        n = -relative_position

        if causal:
            n = torch.clamp(n, min=0)
        else:
            num_buckets //= 2
            sign = (n < 0).long()
            n = torch.abs(n)

        max_exact = num_buckets // 2

        is_small = n < max_exact

        n_clip = torch.clamp(n, min=1)

        val_if_large = max_exact + (
                torch.log(n_clip.float() / max_exact) /
                math.log(max_distance / max_exact) *
                (num_buckets - max_exact)
        ).long()

        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)

        bucket = torch.where(is_small, n, val_if_large)

        if not causal:
            bucket = bucket + sign * num_buckets

        return bucket

    def forward(self, qk_dots):

        i, j = qk_dots.shape[-2:]

        device = qk_dots.device

        q_pos = torch.arange(j - i, j, device=device)
        k_pos = torch.arange(j, device=device)

        rel_pos = k_pos[None, :] - q_pos[:, None]

        buckets = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )

        values = self.relative_attention_bias(buckets)

        bias = rearrange(values, "i j h -> h i j")

        return qk_dots + bias * self.scale


class SelfAttention(nn.Module):

    def __init__(
        self,
        *,
        dim,
        heads=12,
        dim_head=64,
        causal=False,
        dropout=0.,
        max_seq_len=1024
    ):
        super().__init__()

        inner = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim)

        self.dropout = nn.Dropout(dropout)

        self.rel_pos = RelativePositionBias(
            scale=self.scale,
            causal=causal,
            heads=heads
        )

        # кеш маски
        if causal:
            mask = torch.triu(torch.ones(max_seq_len, max_seq_len), 1).bool()
            self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, h, -1).transpose(1, 2)
        k = k.view(b, n, h, -1).transpose(1, 2)
        v = v.view(b, n, h, -1).transpose(1, 2)
        q = q * self.scale
        sim = q @ k.transpose(-1, -2)
        sim = self.rel_pos(sim)
        mask_value = -torch.finfo(sim.dtype).max
        if mask is not None:
            sim = sim.masked_fill(~mask, mask_value)
        if self.causal:
            sim = sim.masked_fill(
                self.causal_mask[:n, :n],
                mask_value
            )
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            context_dim=None,
            heads=12,
            dim_head=64,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, mask=None, context_mask=None):
        b, n, _, h = *x.shape, self.heads
        kv_input = default(context, x)
        q, k, v = self.to_q(x), self.to_k(kv_input), self.to_v(kv_input)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        # sim = q @ k.transpose(-1, -2)
        mask_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            sim = sim.masked_fill(~mask, mask_value)

        if context_mask is not None:
            sim = sim.masked_fill(~context_mask[:, None, :], mask_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        # out = attn @ v.transpose(-1, -2)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_tokens,
            depth,
            heads=12,
            dim_head=64,
            causal=False,
            mlp_mult=4,
            dropout=0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layer = nn.ModuleList([])
        for _ in range(depth):
            self.layer.append(nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(dim=dim, heads=heads, dim_head=dim_head, causal=causal,
                                                      dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim=dim, mult=mlp_mult, dropout=dropout))),
            ]))

        self.final_norm = LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.token_emb(x)
        for attn, mlp in self.layer:
            x = attn(x, mask=mask)
            x = mlp(x)

        x = self.final_norm(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_tokens,
            depth,
            heads=12,
            dim_head=64,
            causal=True,
            mlp_mult=4,
            dropout=0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layer = nn.ModuleList([])
        for _ in range(depth):
            self.layer.append(nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(dim=dim, heads=heads, dim_head=dim_head, causal=causal,
                                                      dropout=dropout))),
                Residual(PreNorm(dim, CrossAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim=dim, mult=mlp_mult, dropout=dropout))),
            ]))

        self.final_norm = LayerNorm(dim)

    def forward(self, x, context, mask=None, context_mask=None):
        x = self.token_emb(x)

        for attn, cross_attn, mlp in self.layer:
            x = attn(x, mask=mask)
            x = cross_attn(x, context=context, mask=mask, context_mask=context_mask)
            x = mlp(x)

        x = self.final_norm(x)

        return x


class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            enc_num_tokens,
            enc_depth,
            enc_heads,
            enc_dim_head,
            enc_mlp_mult,
            dec_num_tokens,
            dec_depth,
            dec_heads,
            dec_dim_head,
            dec_mlp_mult,
            dropout=0.,
            tie_token_emb=True
    ):
        super().__init__()

        # self.embedding = nn.Embedding(enc_num_tokens, dim)
        self.encoder = Encoder(
            dim=dim,
            num_tokens=enc_num_tokens,
            depth=enc_depth,
            heads=enc_heads,
            dim_head=enc_dim_head,
            mlp_mult=enc_mlp_mult,
            dropout=dropout
        )

        self.decoder = Decoder(
            dim=dim,
            num_tokens=dec_num_tokens,
            depth=dec_depth,
            heads=dec_heads,
            dim_head=dec_dim_head,
            mlp_mult=dec_mlp_mult,
            dropout=dropout
        )

        self.to_logits = nn.Linear(dim, dec_num_tokens)

        if tie_token_emb:
            self.encoder.token_emb.weight = self.decoder.token_emb.weight
            self.to_logits.weight = self.decoder.token_emb.weight
            # self.to_logits.weight = self.decoder.token_emb.weight

    def forward(self, src, tgt, mask=None, context_mask=None):
        # x = self.embedding(src)
        x = self.encoder(src, mask=mask)
        x = self.decoder(tgt, x, mask=mask, context_mask=context_mask)
        x = self.to_logits(x)
        return x