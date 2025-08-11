from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from v4_imp import Transformer, tokenize_fn
torch.set_float32_matmul_precision('high')


N_HEADS = 8
MODEL_DIM = N_HEADS * 32
NUM_LAYERS = 16
FF_DIM = MODEL_DIM * 4
DROPOUT = 0.15
BATCH_SIZE = 8
MAX_LEN = 512
NUM_EPOCHS = 50
LR = 1e-4


class TorchLightningModule(LightningModule):
    def __init__(self, model, lr, weight_decay=0):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch['input'])
        loss = self.model.compute_loss(outputs, batch['output'])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch['input'])
        loss = self.model.compute_loss(outputs, batch['output'])
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

model = Transformer(vocab_size=tokenizer.vocab_size,
                    d_model=MODEL_DIM,
                    n_heads=N_HEADS,
                    num_layers=NUM_LAYERS,
                    ff_dim=FF_DIM,
                    dropout=DROPOUT)
model = torch.compile(model, fullgraph=True)
ds = load_dataset("wmt/wmt19", "ru-en")


