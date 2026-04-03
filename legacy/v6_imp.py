import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import lightning as L


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:

            attn = attn.masked_fill(mask == 0, float('-1e9'))

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    # print(seq.size())
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []
        # print(src_seq.type())
        # print(src_seq.size())
        # print(self.src_word_emb)
        enc_output = self.src_word_emb(src_seq)
        # print(self.src_word_emb)
        # print(not torch.any(enc_output.isnan()))
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_tgt_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, tgt_seq, tgt_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(L.LightningModule):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self,
                 n_src_vocab=9960,
                 src_pad_idx=0,
                 d_model=768,
                 d_inner=3072,
                 n_layers=24,
                 n_head=12,
                 d_k=96,
                 d_v=96,
                 dropout=0.1,
                 n_position=256,
                 d_word_vec=None,
                 tgt_pad_idx=None,
                 n_tgt_vocab=None,
                 tgt_emb_prj_weight_sharing=True,
                 emb_src_tgt_weight_sharing=True,
                 scale_emb_or_prj='emb'):

        super().__init__()
        self.save_hyperparameters()

        self.max_seq = n_position
        d_word_vec = d_model if not d_word_vec else d_word_vec
        tgt_pad_idx = src_pad_idx if not tgt_pad_idx else tgt_pad_idx
        n_tgt_vocab = n_src_vocab if not n_tgt_vocab else n_tgt_vocab

        self.src_pad_idx, self.tgt_pad_idx = src_pad_idx, tgt_pad_idx
        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if tgt_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if tgt_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=tgt_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight

        if emb_src_tgt_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

        self.soft = nn.Softmax(dim=-1)


    def forward(self, src_seq, tgt_seq=None):
        tgt_seq = torch.tensor([[self.src_pad_idx] * self.max_seq] * src_seq.size()[0]).to(self.device)
        # try:
        #
        #     b = tgt_seq.type()
        # except Exception as e:
        #     print(e)
        #     tgt_seq = torch.tensor([[self.src_pad_idx] * self.max_seq] * src_seq.size()[0])  # .to(self.device)
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        tgt_mask = get_pad_mask(tgt_seq, self.tgt_pad_idx) & get_subsequent_mask(tgt_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(tgt_seq, tgt_mask, enc_output, src_mask)
        seq_logit = self.tgt_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5
        # print(seq_logit.size())
        # seq_logit = self.soft(seq_logit)
        return seq_logit  # seq_logit.max(-1).indices
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}

    def training_step_full(self, batch, batch_idx):
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.min().item(), param.grad.max().item())
        if batch_idx % 10**4 == 0:
            pass
        input_ids = batch["input"]  # .to(self.device)
        target_ids = batch["output"]  # .to(self.device)
        out = self.generate_next_token(input_ids, target_ids).squeeze()

        loss = F.mse_loss(out.float(), F.one_hot(target_ids, 9960).float(), reduction="mean")  # * self.max_seq / 4  # / self.max_seq / input_ids.size()[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # loss.requires_grad = True
        # print(not torch.any(target_ids.isnan()), not torch.any(input_ids.isnan()), not torch.any(out.isnan()), loss)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input"].squeeze()  # (B, S_src)
        target_ids = batch["output"].squeeze()  # (B, S_tgt)
        # print(target_ids)
        # Teacher forcing: вход декодера = все, кроме последнего токена
        tgt_input = target_ids[:, :-1]  # (B, S_tgt-1)
        tgt_output = target_ids[:, 1:]  # (B, S_tgt-1) ← это цель

        # Маски
        logits = self(input_ids, tgt_input)

        # Считаем CrossEntropyLoss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (B*(S_tgt-1), vocab_size)
            tgt_output.reshape(-1),  # (B*(S_tgt-1),)
            ignore_index=self.tgt_pad_idx,
            reduction='mean'
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss