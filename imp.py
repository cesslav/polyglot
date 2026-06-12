# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.gamma


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.w_gate = nn.Linear(dim, inner_dim, bias=False)
        self.w_up = nn.Linear(dim, inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class ALiBi(nn.Module):
    def __init__(self, heads: int, max_seq_len: int = 1024):
        super().__init__()
        slopes = self._slopes(heads)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        dist = (pos[None, :] - pos[:, None]).abs()
        bias = -dist.unsqueeze(0) * slopes[:, None, None]
        self.register_buffer("_bias", bias, persistent=False)

    @staticmethod
    def _slopes(n: int) -> torch.Tensor:
        def _pow2(m: int):
            start = 2.0 ** (-(2.0 ** -(math.log2(m) - 3)))
            return [start * (start ** i) for i in range(m)]

        if math.log2(n).is_integer():
            return torch.tensor(_pow2(n), dtype=torch.float32)

        n_floor = 2 ** math.floor(math.log2(n))
        extra = _pow2(n_floor * 2)[::2][: n - n_floor]
        return torch.tensor(_pow2(n_floor) + extra, dtype=torch.float32)

    def get_bias(self, q_len: int, k_len: int) -> torch.Tensor:
        return self._bias[:, k_len - q_len : k_len, :k_len]

class MultiQuerySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        causal: bool = False,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self._drop_p = dropout
        inner = heads * dim_head

        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, dim_head, bias=False)
        self.to_v = nn.Linear(dim, dim_head, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.alibi = ALiBi(heads=heads, max_seq_len=max_seq_len)

        if causal:
            cm = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).triu(1)
            self.register_buffer("causal_mask", cm, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        mask = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads

        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(x).unsqueeze(1).expand(b, h, n, -1)
        v = self.to_v(x).unsqueeze(1).expand(b, h, n, -1)

        attn_bias = self.alibi.get_bias(n, n).unsqueeze(0)
        neg_inf = torch.finfo(q.dtype).min

        if self.causal:
            attn_bias = attn_bias.masked_fill(self.causal_mask[:n, :n], neg_inf)

        if mask is not None:
            attn_bias = attn_bias.expand(b, -1, -1, -1).clone()
            attn_bias = attn_bias.masked_fill(~mask[:, None, None, :], neg_inf)

        drop_p = self._drop_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop_p)
        return self.to_out(out.transpose(1, 2).contiguous().view(b, n, -1))


class MultiQueryCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        context_dim = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self._drop_p = dropout

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, dim_head, bias=False)
        self.to_v = nn.Linear(context_dim, dim_head, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask = None,
        context_mask = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.heads
        m = context.shape[1]

        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(context).unsqueeze(1).expand(b, h, m, -1)
        v = self.to_v(context).unsqueeze(1).expand(b, h, m, -1)

        attn_bias = None
        if mask is not None or context_mask is not None:
            neg_inf = torch.finfo(q.dtype).min
            attn_bias = torch.zeros(b, h, n, m, device=x.device, dtype=q.dtype)
            if mask is not None:
                attn_bias = attn_bias.masked_fill(~mask[:, None, :, None], neg_inf)
            if context_mask is not None:
                attn_bias = attn_bias.masked_fill(~context_mask[:, None, None, :], neg_inf)

        drop_p = self._drop_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop_p)
        return self.to_out(out.transpose(1, 2).contiguous().view(b, n, -1))


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_tokens: int,
        depth: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, MultiQuerySelfAttention(
                    dim=dim, heads=heads, dim_head=dim_head,
                    causal=False, dropout=dropout, max_seq_len=max_seq_len,
                ))),
                Residual(PreNorm(dim, FeedForward(dim=dim, mult=mlp_mult, dropout=dropout))),
            ])
            for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask = None,
    ) -> torch.Tensor:
        x = self.token_emb(x)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_tokens: int,
        depth: int,
        heads: int = 8,
        dim_head: int = 64,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, MultiQuerySelfAttention(
                    dim=dim, heads=heads, dim_head=dim_head,
                    causal=True, dropout=dropout, max_seq_len=max_seq_len,
                ))),
                Residual(PreNorm(dim, MultiQueryCrossAttention(
                    dim=dim, heads=heads, dim_head=dim_head, dropout=dropout,
                ))),
                Residual(PreNorm(dim, FeedForward(dim=dim, mult=mlp_mult, dropout=dropout))),
            ])
            for _ in range(depth)
        ])
        self.norm = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask = None,
        context_mask = None,
    ) -> torch.Tensor:
        x = self.token_emb(x)
        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = cross_attn(x, context=context, mask=mask, context_mask=context_mask)
            x = ff(x)
        return self.norm(x)


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
        dropout: float = 0.0,
        max_seq_len: int = 1024,
        tie_token_emb: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            dim=dim, num_tokens=enc_num_tokens, depth=enc_depth,
            heads=enc_heads, dim_head=enc_dim_head, mlp_mult=enc_mlp_mult,
            dropout=dropout, max_seq_len=max_seq_len,
        )
        self.decoder = Decoder(
            dim=dim, num_tokens=dec_num_tokens, depth=dec_depth,
            heads=dec_heads, dim_head=dec_dim_head, mlp_mult=dec_mlp_mult,
            dropout=dropout, max_seq_len=max_seq_len,
        )
        self.to_logits = nn.Linear(dim, dec_num_tokens, bias=False)
        self.to_logits.weight = self.decoder.token_emb.weight

        if tie_token_emb:
            if enc_num_tokens != dec_num_tokens:
                raise ValueError(
                    f"tie_token_emb=True требует enc_num_tokens == dec_num_tokens, "
                    f"получено {enc_num_tokens} и {dec_num_tokens}"
                )
            self.encoder.token_emb.weight = self.decoder.token_emb.weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask = None,
        tgt_mask = None,
    ) -> torch.Tensor:
        context = self.encoder(src, mask=src_mask)
        x = self.decoder(tgt, context, mask=tgt_mask, context_mask=src_mask)
        return self.to_logits(x)


if __name__ == "__main__":
    model = Transformer(
        dim=256,
        enc_num_tokens=8000,
        enc_depth=4,
        enc_heads=8,
        enc_dim_head=32,
        enc_mlp_mult=4,
        dec_num_tokens=8000,
        dec_depth=4,
        dec_heads=8,
        dec_dim_head=32,
        dec_mlp_mult=4,
        dropout=0.1,
        max_seq_len=512,
    )

    src = torch.randint(0, 8000, (2, 64))
    tgt = torch.randint(0, 8000, (2, 48))
    src_mask = torch.ones(2, 64, dtype=torch.bool)
    tgt_mask = torch.ones(2, 48, dtype=torch.bool)

    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print(f"logits shape: {logits.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")