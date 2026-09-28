# This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Нормализация по среднеквадратичному значению (RMS, Root Mean Square) с обучаемым коэффициентом масштабирования gamma.
            Конструктор:
                dim (int) - размерность нормализуемого последнего измерения;
                eps (float) - слагаемое численной стабильности.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Нормирует вход по RMS и масштабирует гаммой.
                Входы:
                    x (Tensor [..., dim]) - входные представления.
                Выходы:
                    output (Tensor [..., dim]) - нормализованные и масштабированные представления.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.gamma


class Residual(nn.Module):
    """Обёртка остаточного соединения: применяет функцию к входу и складывает результат с самим входом.
            Конструктор:
                fn (nn.Module) - модуль-преобразование.
    """
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Применяет внутреннее преобразование и прибавляет исходный тензор.
                Входы:
                    x (Tensor) - вход;
                    kwargs (dict) - дополнительные аргументы для fn (например, mask).
                Выходы:
                    output (Tensor) - fn(x) + x.
        """
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    """Пре-нормализация: сначала RMSNorm, затем преобразование fn (стиль pre-LN Transformer).
            Конструктор:
                dim (int) - размерность;
                fn (nn.Module) - преобразование после нормализации.
    """
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Нормирует вход и прогоняет через внутреннее преобразование.
                Входы:
                    x (Tensor) - вход;
                    kwargs (dict) - дополнительные аргументы, пробрасываются в fn.
                Выходы:
                    output (Tensor) - результат fn(norm(x)).
        """
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Позиционно-независимый MLP в стиле GLU (gate/up/down-проекции, активация SiLU).
            Конструктор:
                dim (int) - размерность входа/выхода;
                mult (int) - множитель внутренней размерности;
                dropout (float) - вероятность отбрасывания.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.w_gate = nn.Linear(dim, inner_dim, bias=False)
        self.w_up = nn.Linear(dim, inner_dim, bias=False)
        self.w_down = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Вычисляет GLU-преобразование: down(SiLU(gate(x)) * up(x)) с dropout.
                Входы:
                    x (Tensor) - тензор (..., dim).
                Выходы:
                    output (Tensor) - результат той же формы (..., dim).
        """
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class ALiBi(nn.Module):
    """Bias-матрицы для ALiBi (Attention with Linear Biases): линейное расстояние от сужением на каждый head, без позиционных эмбеддингов.
            Конструктор:
                heads (int) - число attention-голов;
                max_seq_len (int) - максимальная длина последовательности.
    """
    def __init__(self, heads: int, max_seq_len: int = 1024):
        super().__init__()
        slopes = self._slopes(heads)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        dist = (pos[None, :] - pos[:, None]).abs()
        bias = -dist.unsqueeze(0) * slopes[:, None, None]
        self.register_buffer("_bias", bias, persistent=False)

    @staticmethod
    def _slopes(n: int) -> torch.Tensor:
        """Вычисляет вектор наклонностей (slopes) для n голов по геометрической прогрессии 2^(-(2^-(m-4-i))).
                Входы:
                    n (int) - число голов (не обязательно степень двойки).
                Выходы:
                    slopes (Tensor (n,)) - наклонности для каждой головы.
        """
        def _pow2(m: int):
            """Возвращает список из m наклонностей по формуле 2^(-(2^-(m-4-i))) для i = 0..m-1 (m — степень двойки).
                    Входы:
                        m (int) - число наклонностей (степень двойки).
                    Выходы:
                        slopes (list[float]) - список наклонностей.
            """
            start = 2.0 ** (-(2.0 ** -(math.log2(m) - 3)))
            return [start * (start ** i) for i in range(m)]

        if math.log2(n).is_integer():
            return torch.tensor(_pow2(n), dtype=torch.float32)

        n_floor = 2 ** math.floor(math.log2(n))
        extra = _pow2(n_floor * 2)[::2][: n - n_floor]
        return torch.tensor(_pow2(n_floor) + extra, dtype=torch.float32)

    def get_bias(self, q_len: int, k_len: int) -> torch.Tensor:
        """Вырезает подматрицу ALiBi-смещений для текущих длин запроса и ключей.
                Входы:
                    q_len (int) - длина последовательности запросов;
                    k_len (int) - длина последовательности ключей.
                Выходы:
                    bias (Tensor (heads, q_len, k_len)) - матрица смещений для attention.
        """
        return self._bias[:, k_len - q_len : k_len, :k_len]

class MultiQuerySelfAttention(nn.Module):
    """Самовнимание в схеме multi-query attention: много query-голов, по одной общей K/V-паре, с ALiBi-смещениями.
            Конструктор:
                dim (int) - размерность;
                heads (int) - число query-голов;
                dim_head (int) - размерность головы;
                causal (bool) - использовать ли каузальную (нижнетреугольную) маску;
                dropout (float) - dropout внимания;
                max_seq_len (int) - максимальная длина последовательности для ALiBi.
    """
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
        """Вычисляет multi-query самовнимание с ALiBi и опциональными каузальной и pad-масками.
                Входы:
                    x (Tensor) - (batch, seq, dim), входная последовательность;
                    mask (Tensor|None) - булева маска (batch, seq), True — валидный токен.
                Выходы:
                    output (Tensor) - (batch, seq, dim), результат внимания.
        """
        b, n, _ = x.shape
        h = self.heads

        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(x).unsqueeze(1)
        v = self.to_v(x).unsqueeze(1)

        attn_bias = self.alibi.get_bias(n, n).unsqueeze(0)
        neg_inf = torch.finfo(q.dtype).min

        if self.causal:
            attn_bias = attn_bias.masked_fill(self.causal_mask[:n, :n], neg_inf)

        if mask is not None:
            attn_bias = attn_bias.masked_fill(~mask[:, None, None, :], neg_inf)

        drop_p = self._drop_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=drop_p)
        return self.to_out(out.transpose(1, 2).contiguous().view(b, n, -1))


class MultiQueryCrossAttention(nn.Module):
    """Кросс-внимание в схеме multi-query attention: query из декодера, общие K/V из контекста кодера.
            Конструктор:
                dim (int) - размерность декодера;
                context_dim (int|None) - размерность контекста (по умолчанию = dim);
                heads (int) - число query-голов;
                dim_head (int) - размерность головы;
                dropout (float) - dropout внимания.
    """
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
        """Вычисляет кросс-внимание «декодер -> контекст кодера» с pad-масками обеих сторон.
                Входы:
                    x (Tensor) - (batch, n, dim), последовательность декодера;
                    context (Tensor) - (batch, m, context_dim), контекст кодера;
                    mask (Tensor|None) - маска x (batch, n);
                    context_mask (Tensor|None) - маска контекста (batch, m).
                Выходы:
                    output (Tensor) - (batch, n, dim), результат кросс-внимания.
        """
        b, n, _ = x.shape
        h = self.heads
        m = context.shape[1]

        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(context).unsqueeze(1)
        v = self.to_v(context).unsqueeze(1)

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
    """Кодер seq2seq-модели: эмбеддинги токенов + стек блоков «self-attention (ALiBi) + FFN» с pre-norm и residual, финальный RMSNorm.
            Конструктор:
                dim (int) - размерность скрытого слоя;
                num_tokens (int) - размер словаря;
                depth (int) - число слоёв;
                heads (int) - число голов;
                dim_head (int) - размерность головы;
                mlp_mult (int) - множитель MLP;
                dropout (float) - dropout;
                max_seq_len (int) - максимальная длина для ALiBi.
    """
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
        """Кодирует последовательность токенов в контекстные представления.
                Входы:
                    x (Tensor) - (batch, seq), id токенов;
                    mask (Tensor|None) - булева маска (batch, seq), True — валидный токен.
                Выходы:
                    memory (Tensor) - (batch, seq, dim), память кодера.
        """
        x = self.token_emb(x)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return self.norm(x)


class Decoder(nn.Module):
    """Декодер seq2seq-модели: эмбеддинги + стек блоков «каузальное self-attention + cross-attention + FFN», финальный RMSNorm.
            Конструктор:
                dim (int) - размерность скрытого слоя;
                num_tokens (int) - размер словаря;
                depth (int) - число слоёв;
                heads (int) - число голов;
                dim_head (int) - размерность головы;
                mlp_mult (int) - множитель MLP;
                dropout (float) - dropout;
                max_seq_len (int) - максимальная длина для ALiBi.
    """
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
        """Декодирует последовательность с учётом контекста кодера.
                Входы:
                    x (Tensor) - (batch, tgt_seq), id входных токенов декодера;
                    context (Tensor) - (batch, src_seq, dim), память кодера;
                    mask (Tensor|None) - маска x;
                    context_mask (Tensor|None) - маска контекста.
                Выходы:
                    output (Tensor) - (batch, tgt_seq, dim), представления декодера.
        """
        x = self.token_emb(x)
        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = cross_attn(x, context=context, mask=mask, context_mask=context_mask)
            x = ff(x)
        return self.norm(x)


class Transformer(nn.Module):
    """Полная seq2seq-модель «Encoder + Decoder + логит-головка» с общим словарём и (опционально) связанными эмбеддингами.
            Конструктор:
                dim (int) - размерность скрытого слоя;
                enc_num_tokens/dec_num_tokens (int) - размеры словарей кодера/декодера;
                enc_depth/dec_depth (int) - число слоёв;
                enc_heads/dec_heads (int) - число голов;
                enc_dim_head/dec_dim_head (int) - размерность головы;
                enc_mlp_mult/dec_mlp_mult (int) - множители MLP;
                dropout (float) - dropout;
                max_seq_len (int) - максимальная длина для ALiBi;
                tie_token_emb (bool) - общие ли веса эмбеддингов кодера и декодера.
    """
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
        """Прогоняет исходную и целевую последовательности через кодер и декодер, выдаёт logits.
                Входы:
                    src (Tensor) - (batch, src_seq), id исходных токенов;
                    tgt (Tensor) - (batch, tgt_seq), id целевых токенов;
                    src_mask (Tensor|None) - маска исходного текста;
                    tgt_mask (Tensor|None) - маска целевого текста.
                Выходы:
                    logits (Tensor) - (batch, tgt_seq, dec_num_tokens), logits по словарю на каждую позицию.
        """
        context = self.encoder(src, mask=src_mask)
        x = self.decoder(tgt, context, mask=tgt_mask, context_mask=src_mask)
        return self.to_logits(x)


if __name__ == "__main__":
    model = Transformer(
        dim=512,
        enc_num_tokens=48000,
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

    src = torch.randint(0, 48000, (2, 64))
    tgt = torch.randint(0, 48000, (2, 48))
    src_mask = torch.ones(2, 64, dtype=torch.bool)
    tgt_mask = torch.ones(2, 48, dtype=torch.bool)

    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print(f"logits shape: {logits.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    print("This file is distributed under the open license AGPLv3, source code: https://github.com/cesslav/polyglot.")