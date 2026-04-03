import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_scheduler
import pytorch_lightning as pl


# ===================== DataModule =====================
class WMT19DataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size=8, max_len=128):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        self.dataset = load_dataset("wmt19", "de-en")

        def preprocess(examples):
            src = examples["translation"]["de"]
            tgt = examples["translation"]["en"]
            model_inputs = self.tokenizer(
                src, max_length=self.max_len, truncation=True
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    tgt, max_length=self.max_len, truncation=True
                )
            model_inputs["target_ids"] = labels["input_ids"]
            return model_inputs

        self.dataset = self.dataset.map(preprocess, batched=True, remove_columns=["translation"])

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.batch_size)


# ===================== LightningModule =====================
class TranslationModel(pl.LightningModule):
    def __init__(self, model_name, lr=5e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=10000
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


# ===================== Запуск обучения =====================
if __name__ == "__main__":
    model_name = "facebook/wmt19-de-en"
    dm = WMT19DataModule(model_name, batch_size=4)
    model = TranslationModel(model_name)

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices="auto",
        precision="16-mixed"
    )
    trainer.fit(model, dm)
