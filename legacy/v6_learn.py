from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader
from datasets import load_from_disk
from datetime import timedelta
from os import listdir
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from v71_imp import Transformer
from torch import nn

set_float32_matmul_precision('high')


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -1, 1)


def main(batch_size, num_epochs, train_loader):
    delta = timedelta(hours=4)
    point = listdir("../proto/")

    checkpoint_callback = ModelCheckpoint(dirpath="./proto/",
                                          # every_n_train_steps=int(len(train_loader)/4),
                                          train_time_interval=delta,
                                          mode="min"
                                          )

    if point: model = Transformer.load_from_checkpoint(f"./proto/{point[0]}")
    else: model = Transformer().apply(init_weights)

    trainer = L.Trainer(devices="auto", accelerator="gpu",  # strategy="ddp",
                        accumulate_grad_batches=batch_size * 4,
                        callbacks=[
                                   checkpoint_callback,
                                   # StochasticWeightAveraging(swa_lrs=1e-2)
                        ],
                        max_epochs=num_epochs, min_epochs=1,
                        precision="bf16-true", enable_progress_bar=True,
                        # gradient_clip_val=0.1, gradient_clip_algorithm="norm"
                        )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    num_epochs = 1000
    batch_size = 2
    trainset = load_from_disk("./sources/wmt19_s256_1").take(100)
    print(len(trainset))
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    trainset = 0
    main(batch_size, num_epochs, train_loader)
