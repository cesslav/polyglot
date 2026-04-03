import torch
from torch import nn
import random
import lightning as L
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src : [sen_len, batch_size]
        embedded = self.dropout(self.embedding(src))

        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch_size]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        input = input.unsqueeze(0)
        # input : [1, ,batch_size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell


class Seq2Seq(L.LightningModule):
    def __init__(self,
                 dim=9960,
                 emb_dim=256,
                 hid_dim=1024,
                 n_layers=48,
                 dropout=0.1):
        super().__init__()

        self.encoder = Encoder(dim, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = Decoder(dim, emb_dim, hid_dim, n_layers, dropout)
        self.soft = nn.Softmax(dim=-1)
        # self.device = device

        assert self.encoder.hid_dim == self.encoder.hid_dim, \
            'hidden dimensions of encoder and decoder must be equal.'
        assert self.decoder.n_layers == self.decoder.n_layers, \
            'n_layers of encoder and decoder must be equal.'

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        self.loss_func = nn.CrossEntropyLoss()
        # src = [seq_len, batch_size]
        # trg = [seq_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> token.
        input = trg[0, :]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)

            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force
            input = trg[t] if teacher_force else top1
        outputs = self.soft(outputs)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-7)
        return optimizer

    def training_step(self, batch, batch_idx):
        src = batch["input"].squeeze()
        trg = batch["output"].squeeze()

        output = self(src, trg)
        output_dim = output.shape[-1]

        # transfrom our output : slice off the first column, and flatten the output into 2 dim.
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        # trg = [(trg_len-1) * batch_size]
        # output = [(trg_len-1) * batch_size, output_dim]
        hot = F.one_hot(trg, num_classes=9960)
        # print("")
        # print(output)
        # print(hot)
        loss = self.loss_func(output.float(), hot.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(loss)

        return loss