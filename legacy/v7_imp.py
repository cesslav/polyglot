import torch
import torch.nn.functional as Fun
from torch import nn
import math



class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) #  if bias else None

    def forward(self, input):
        return Fun.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.dim_embedding)
        self.wtp = nn.Embedding(config.block_size, config.dim_embedding)

    def forward(self, x):
        x = self.wte(x).to(self.config.device)
        position_ids = (
            torch.arange(self.config.block_size).unsqueeze(0).repeat(x.size(0), 1)
        ).to(self.config.device)
        position_embeddings = self.wtp(position_ids)
        x = x + position_embeddings

        return x


class ProcessingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.denselayer1 = nn.Linear(config.dim_embedding, config.dim_inner_layer)
        self.denselayer2 = nn.Linear(config.dim_inner_layer, config.dim_embedding)
        self.layernorm1 = LayerNorm(config.dim_embedding, bias=config.bias)
        self.layernorm2 = LayerNorm(config.dim_embedding, bias=config.bias)

    def forward(self, x):
        x_in = self.layernorm1(x)
        x = self.denselayer1(x_in)
        x = self.denselayer2(x) + x_in
        x = self.layernorm2(x)
        return x


def split_qkv(x, q, k, v, dim_embedding, n_head):
    (
        B,
        T,
        C,
    ) = x.size()

    """Function for attention calculation which splits k, q and v down to batch_size, 
    number_heads, block size, dimension_embedding/number_heads"""
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
    return q, k, v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim_embedding % config.n_head == 0

        # set embedding and head sizes
        self.n_head = config.n_head
        self.dim_embedding = config.dim_embedding

        self.c_attn = nn.Linear(
            config.dim_embedding, 3 * config.dim_embedding, bias=config.bias
        )
        self.c_proj = nn.Linear(
            config.dim_embedding, config.dim_embedding, bias=config.bias
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

    def attention_func(self, x, q, k, v, mask=False):
        (
            B,
            T,
            C,
        ) = x.size()

        if self.flash:
            if mask == True:
                causal_status = True
            elif mask == False:
                causal_status = False
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=causal_status,
            )

        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask == True:
                att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = Fun.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y

    def forward(self, x):
        q, k, v = self.c_attn(x).split(self.dim_embedding, dim=2)

        q, k, v = split_qkv(x, q, k, v, self.dim_embedding, self.n_head)

        y = self.attention_func(x, q, k, v, mask=False)

        y = self.resid_dropout(self.c_proj(y))

        return y


class MaskedMultiHeadAttention(MultiHeadAttention):
    """MaskedMultiHeadAttention class inherits from MultiHeadAttention"""

    def __init__(self, config):
        super().__init__(config)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )  # (B, nh, T, hs)

    def forward(self, x):
        q, k, v = self.c_attn(x).split(self.dim_embedding, dim=2)

        q, k, v = split_qkv(x, q, k, v, self.dim_embedding, self.n_head)

        y = self.attention_func(x, q, k, v, mask=True)

        y = self.resid_dropout(self.c_proj(y))

        return y


class EncoderDecoderAttention(MultiHeadAttention):
    def __init__(self, config):
        super().__init__(config)

        self.c_attn_en = nn.Linear(
            config.dim_embedding, 2 * config.dim_embedding, bias=config.bias
        )
        self.c_attn = nn.Linear(
            config.dim_embedding, config.dim_embedding, bias=config.bias
        )

    def forward(self, x, e):
        k, v = self.c_attn_en(e).split(self.dim_embedding, dim=2)
        q = self.c_attn(x)

        q, k, v = split_qkv(x, q, k, v, self.dim_embedding, self.n_head)

        y = self.attention_func(x, q, k, v, mask=True)

        y = self.resid_dropout(self.c_proj(y))

        return y


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.attention_encoder = MultiHeadAttention(config)

        self.encoder_processing_layer = ProcessingLayer(config)

    def forward(self, x):

        x = self.attention_encoder(x) + x

        x = self.encoder_processing_layer(x)

        return x


# The Decoder class


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.masked_attention = MaskedMultiHeadAttention(config)

        self.layernorm = LayerNorm(config.dim_embedding, bias=config.bias)

        self.encoder_decoder_attn = EncoderDecoderAttention(config)

        self.decoder_processing_layer = ProcessingLayer(config)

    def forward(self, x, y):
        y = self.masked_attention(y) + y

        y = self.layernorm(y)

        y = self.encoder_decoder_attn(y, x) + y

        y = self.decoder_processing_layer(y)

        return y


class TransformerL(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        self.encoder_embed = Embedding(config)
        self.decoder_embed = Embedding(config)

        self.encoders = nn.ModuleList([Encoder(config) for _ in range(config.n_layers)])
        self.decoders = nn.ModuleList([Decoder(config) for _ in range(config.n_layers)])

        self.final_layer = nn.Linear(config.dim_embedding, config.vocab_size)

    def forward(self, x, y):
        x = self.encoder_embed(x)
        y = self.decoder_embed(y)

        for encoder in self.encoders:
            x = encoder(x)

        for decoder in self.decoders:
            y = decoder(x, y)

        y = self.final_layer(y)

        return y