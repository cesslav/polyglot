import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), self.gamma, self.beta)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.w1 = nn.Linear(dim, inner_dim)
        self.w2 = nn.Linear(dim, inner_dim)
        self.out = nn.Linear(inner_dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.dropout1(x)
        x = self.dropout2(self.out(x))
        return x


class RelativePositionBias(nn.Module):
    def __init__(
            self,
            scale: float,
            causal: bool,
            num_buckets: int = 32,
            max_distance: int = 128,
            heads: int = 12,
            max_seq_len: int = 1024,
    ):
        super().__init__()
        self.scale = scale
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

        rows = torch.arange(max_seq_len)
        cols = torch.arange(max_seq_len)
        rel = cols[None, :] - rows[:, None]
        buckets = self._relative_position_bucket(
            rel, causal=causal,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )
        self.register_buffer("_buckets", buckets, persistent=False)


    @staticmethod
    def _relative_position_bucket(
            relative_position: torch.Tensor,
            causal: bool = True,
            num_buckets: int = 32,
            max_distance: int = 128,
    ) -> torch.Tensor:
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
        val_large = max_exact + (
                torch.log(n_clip.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).long()
        val_large = torch.clamp(val_large, max=num_buckets - 1)
        bucket = torch.where(is_small, n, val_large)
        if not causal:
            bucket = bucket + sign * num_buckets
        return bucket

    def get_bias(self, q_len: int, k_len: int) -> torch.Tensor:
        buckets = self._buckets[k_len - q_len: k_len, :k_len]
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1) * self.scale

    def forward(self, qk_dots: torch.Tensor) -> torch.Tensor:
        i, j = qk_dots.shape[-2:]
        bias = self.get_bias(i, j)
        return qk_dots + bias


class SelfAttention(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            heads: int = 12,
            dim_head: int = 64,
            causal: bool = False,
            dropout: float = 0.,
            max_seq_len: int = 1024,
    ):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        self._dropout_p = dropout

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

        self.rel_pos = RelativePositionBias(
            scale=self.scale,
            causal=causal,
            heads=heads,
            max_seq_len=max_seq_len,
        )

        if causal:
            mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).triu(1)
            self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, h, -1).transpose(1, 2)
        k = k.view(b, n, h, -1).transpose(1, 2)
        v = v.view(b, n, h, -1).transpose(1, 2)

        attn_bias = self.rel_pos.get_bias(n, n).unsqueeze(0)
        neg_inf = torch.finfo(q.dtype).min

        if mask is not None:
            attn_bias = attn_bias.expand(b, -1, -1, -1).clone()
            attn_bias = attn_bias.masked_fill(~mask, neg_inf)

        if self.causal:
            causal_m = self.causal_mask[:n, :n]
            attn_bias = attn_bias.masked_fill(causal_m, neg_inf)

        drop_p = self._dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=drop_p,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            context_dim: Optional[int] = None,
            heads: int = 12,
            dim_head: int = 64,
            dropout: float = 0.,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self._dropout_p = dropout

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        kv_input = context if context is not None else x
        m = kv_input.shape[1]

        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(kv_input).view(b, m, h, -1).transpose(1, 2)
        v = self.to_v(kv_input).view(b, m, h, -1).transpose(1, 2)

        attn_bias: Optional[torch.Tensor] = None
        neg_inf = torch.finfo(q.dtype).min

        if mask is not None or context_mask is not None:
            attn_bias = torch.zeros(b, h, n, m, device=x.device, dtype=q.dtype)
            if mask is not None:
                attn_bias = attn_bias.masked_fill(~mask, neg_inf)
            if context_mask is not None:
                cm = context_mask[:, None, None, :]
                attn_bias = attn_bias.masked_fill(~cm, neg_inf)

        drop_p = self._dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=drop_p,
            scale=self.scale,
        )

        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class Encoder(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            num_tokens: int,
            depth: int,
            heads: int = 12,
            dim_head: int = 64,
            causal: bool = False,
            mlp_mult: int = 4,
            dropout: float = 0.,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layer = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(
                    dim=dim, heads=heads, dim_head=dim_head,
                    causal=causal, dropout=dropout,
                ))),
                Residual(PreNorm(dim, FeedForward(
                    dim=dim, mult=mlp_mult, dropout=dropout,
                ))),
            ])
            for _ in range(depth)
        ])
        self.final_norm = LayerNorm(dim)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_emb(x)
        for attn, mlp in self.layer:
            x = attn(x, mask=mask)
            x = mlp(x)
        return self.final_norm(x)


class Decoder(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            num_tokens: int,
            depth: int,
            heads: int = 12,
            dim_head: int = 64,
            causal: bool = True,
            mlp_mult: int = 4,
            dropout: float = 0.,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layer = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(
                    dim=dim, heads=heads, dim_head=dim_head,
                    causal=causal, dropout=dropout,
                ))),
                Residual(PreNorm(dim, CrossAttention(
                    dim=dim, heads=heads, dim_head=dim_head, dropout=dropout,
                ))),
                Residual(PreNorm(dim, FeedForward(
                    dim=dim, mult=mlp_mult, dropout=dropout,
                ))),
            ])
            for _ in range(depth)
        ])
        self.final_norm = LayerNorm(dim)

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_emb(x)
        for attn, cross_attn, mlp in self.layer:
            x = attn(x, mask=mask)
            x = cross_attn(x, context=context, mask=mask, context_mask=context_mask)
            x = mlp(x)
        return self.final_norm(x)


class Transformer(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            enc_num_tokens: int,
            enc_depth: int,
            enc_heads: int,
            enc_dim_head: int,
            enc_mlp_mult: int,
            dec_num_tokens: int,
            dec_depth: int,
            dec_heads: int,
            dec_dim_head: int,
            dec_mlp_mult: int,
            dropout: float = 0.,
            tie_token_emb: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            dim=dim, num_tokens=enc_num_tokens, depth=enc_depth,
            heads=enc_heads, dim_head=enc_dim_head, mlp_mult=enc_mlp_mult,
            dropout=dropout,
        )
        self.decoder = Decoder(
            dim=dim, num_tokens=dec_num_tokens, depth=dec_depth,
            heads=dec_heads, dim_head=dec_dim_head, mlp_mult=dec_mlp_mult,
            dropout=dropout,
        )
        self.to_logits = nn.Linear(dim, dec_num_tokens)

        if tie_token_emb:
            self.encoder.token_emb.weight = self.decoder.token_emb.weight
            self.to_logits.weight = self.decoder.token_emb.weight

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.encoder(src, mask=mask)
        x = self.decoder(tgt, x, mask=mask, context_mask=context_mask)
        return self.to_logits(x)


if __name__ == "__main__":
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")