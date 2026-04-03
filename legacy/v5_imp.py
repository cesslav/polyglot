import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class ScaledDotProductAttention(nn.Module):

    def __init__(self, scale, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q * self.scale, k.transpose(2, 3))

        attn = attn.masked_fill(mask == 0, -torch.finfo(attn.dtype).min) if mask is not None else attn

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

        self.embed_qs = nn.Linear(d_model, d_k * num_heads)
        self.embed_ks = nn.Linear(d_model, d_k * num_heads)
        self.embed_vs = nn.Linear(d_model, d_v * num_heads)

        self.attn = ScaledDotProductAttention(scale=d_k ** (- 0.5))
        self.fc = nn.Linear(num_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-4)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, num_heads = self.d_k, self.d_v, self.num_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.embed_qs(q).view(batch_size, len_q, num_heads, d_k).transpose(1, 2)
        k = self.embed_qs(k).view(batch_size, len_k, num_heads, d_k).transpose(1, 2)
        v = self.embed_qs(v).view(batch_size, len_v, num_heads, d_v).transpose(1, 2)

        mask = mask.unsqueeze(1) if mask is not None else mask

        q, attn = self.attn(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.norm(residual + q)

        return q, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


def padding_mask(seq_k, seq_q):
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, seq_q.size(1), -1)
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (- (torch.tensor(10000.0).log() / d_model)))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hid, num_heads, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_self_attn


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_hid, num_heads, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(num_heads, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_self_attn = self.self_attn(dec_input, dec_input, dec_input, mask=self_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):

    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_heads, d_k, d_v, d_model, d_hid, pad_idx,
                 dropout=0.1, max_len=200.):
        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_word_vec, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_hid, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-4)

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_self_attn_list = []
        enc_output = self.src_word_emb(src_seq)
        enc_output = self.pos_enc(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(enc_output, self_attn_mask=src_mask)
            enc_self_attn_list += [enc_self_attn] if return_attns else []

        return enc_output, enc_self_attn_list if return_attns else enc_output,


class Decoder(nn.Module):

    def __init__(self, n_tgt_vocab, d_word_vec, n_layers, n_heads, d_k, d_v, d_model, d_hid, pad_idx,
                 max_len=200, dropout=0.1):
        super(Decoder, self).__init__()

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_word_vec, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_hid, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=1e-4)

    def forward(self, tgt_seq, tgt_mask, enc_output, src_mask, return_attns=False):
        dec_self_attn_list, dec_enc_attn_list = [], []

        dec_output = self.dropout(self.pos_enc(self.tgt_word_emb(tgt_seq)))
        dec_output = self.norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn, dec_enc_attn = dec_layer(dec_output, enc_output,
                                                                self_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask)
            dec_self_attn_list += [dec_self_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        return dec_output, dec_self_attn_list, dec_enc_attn_list if return_attns else dec_output,



class Transformer(L.LightningModule):

    def __init__(self, 
                 n_src_vocab=9960,
                 src_pad_idx=1,
                 d_word_vec=768,
                 d_model=768,
                 d_hid=1644,
                 n_layers=16,
                 n_heads=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1, 
                 max_len=256,
                 tgt_pad_idx=0,
                 n_tgt_vocab=0,
                 tgt_emb_prj_weight_sharing=True,
                 emb_src_tgt_weight_sharing=True):
        super().__init__()
        # self.bos = torch.tensor([[src_pad_idx] * max_len] * batch_size).to(self.device)
        n_tgt_vocab = n_tgt_vocab if n_tgt_vocab else n_src_vocab
        tgt_pad_idx = tgt_pad_idx if tgt_pad_idx else src_pad_idx
        assert d_model == d_word_vec, "To facilitate the resudual connections, the dimensions of all module outputs shall be the same."
        self.max_seq_len = max_len
        self.src_pad_idx, self.tgt_pad_idx = src_pad_idx, tgt_pad_idx
        self.x_logit_scale = (d_model ** (- 0.5))
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_heads, d_k, d_v, d_model, d_hid, pad_idx=src_pad_idx, max_len=max_len, dropout=dropout)
        self.decoder = Decoder(n_tgt_vocab, d_word_vec, n_layers, n_heads, d_k, d_v, d_model, d_hid, pad_idx=tgt_pad_idx, max_len=max_len, dropout=dropout)
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        self.init_weights(tgt_emb_prj_weight_sharing, emb_src_tgt_weight_sharing)

    def init_weights(self, tgt_emb_prj_weight_sharing, emb_src_tgt_weight_sharing):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, 0, 1)

        if tgt_emb_prj_weight_sharing:
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight

        if emb_src_tgt_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq):  # , tgt_seq
        # print(src_seq.size(), tgt_seq.size())
        tgt_seq = torch.tensor([[self.src_pad_idx] * self.max_seq_len] * src_seq.size()[0]).to(self.device)
        # print(not torch.any(tgt_seq.isnan()))
        src_seq.to(self.device)
        tgt_seq.to(self.device)
        src_mask = self._get_pad_mask(src_seq, self.src_pad_idx)
        # print(src_mask)
        tgt_mask = self._get_pad_mask(tgt_seq, self.tgt_pad_idx) & self._get_subsequent_mask(tgt_seq)
        # print(tgt_mask)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        # print(not torch.any(enc_output.isnan()))
        dec_output, *_ = self.decoder(tgt_seq.to(self.device), tgt_mask.to(self.device), enc_output.to(self.device), src_mask.to(self.device))
        # print(not torch.any(dec_output.isnan()))
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
        # print(seq_logit.size())
        out = seq_logit  # .max(dim=2).indices
        return out

    @staticmethod
    def _get_pad_mask(seq, pad_idx):
        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def _get_subsequent_mask(seq):
        len_seq = seq.size(-1)
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_seq, len_seq), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-7)
        return optimizer

    def training_step(self, batch, batch_idx):
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.min().item(), param.grad.max().item())
        input_ids = batch["input"].to(self.device)
        target_ids = F.one_hot(batch["output"], 9960).to(self.device)
        out = self.forward(input_ids).squeeze()

        loss = F.mse_loss(out.float(), target_ids.float(), reduction="sum")  #  / 8 / input_ids.size()[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # loss.requires_grad = True
        # print(not torch.any(target_ids.isnan()), not torch.any(input_ids.isnan()), not torch.any(out.isnan()), loss)

        return loss

