import sys
from time import sleep
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as Fun
from torch import nn
import torch.optim as optim
import v7_imp_mod as Tr
from v7_conf import TransformerConfig
from tqdm import tqdm
# Initialize configuration
config = TransformerConfig()

# Initialize model
model = Tr.TransformerL(config).to(config.device)

# Set up Tensorboard writer
writer = SummaryWriter()

# german, englishImport the training and mixed48k datasets and convert them into data loaders
# _ = load_from_disk("./sources/wmt19_s256_full").take(500)
train_dataset = load_from_disk("../sources/wmt19_s256_full").select(range(500))
test_dataset = load_from_disk("./sources/wmt19_s256_val")

# train_dataset = torch.load("train_dataset.pt", weights_only=False)
# test_dataset = torch.load("test_dataset.pt", weights_only=False)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

# Allow gradients to be computed
torch.set_grad_enabled(True)

# Use the Adam optimizer
optimizer = optim.Adam(model.parameters(), 1e-4)
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
torch.autograd.set_detect_anomaly(True)
mode = "train"



def save_checkpoint(model, optimizer, save_path, loss, config):
    """function to save checkpoints_1024 on the model weights, the optimiser state and epoch"""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "loss": loss
        },
        save_path,
    )


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        if m.bias is not None:
            nn.init.constant_(m.weight, 1.0)


# If a saved model is input load the model and the optimizer state, else start from 0
# if sys.argv[1] == "new_model":
#     epoch_start = 0
# elif sys.argv[1] is not None:
#     state = torch.load(sys.argv[1])
#     model.load_state_dict(state["model_state_dict"])
#     optimizer.load_state_dict(state["optimizer_state_dict"])
#     epoch_start = state["epoch"]


if mode == "train":
    model.apply(init_weights)
elif mode == "continue":
    state = torch.load("checkpoints_1024/model_post_2", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    # epoch_start = state["epoch"]


print(sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1000000000)
print(torch.cuda.memory_allocated() / 1024**3)

print(model)

# The training loop (The + 1 is get the numbers we want from range)
for epoch in range(1, config.epochs + 1):
    sleep(0.05)
    model.train()
    sum_loss = 0
    count = 0
    loop = tqdm(train_dataloader)
    nan_counter = 0
    for batch in loop:
        german = batch["output"].to(config.device).squeeze()
        english = batch["input"].to(config.device).squeeze()
        output = batch["output"].to(config.device).squeeze()


        # german = batch[0].to(config.device)
        # english = batch[1].to(config.device)
        # output = batch[2].to(config.device)
        # print(german, german.size())
        # print(english, english.size())
        # print(output, output.size())

        # clear any existing gradients to compute a new pone
        optimizer.zero_grad()

        # Generates a predicted translation
        yhat = model(english, german)

        # Calculates the cross entropy between the prediction and the target, ignoring the 0s
        loss = Fun.cross_entropy(
            yhat.view(-1, yhat.size(-1)),
            output.view(-1),
            ignore_index=0,
        )
        if torch.isnan(loss).any() or torch.isnan(german).any() or torch.isnan(english).any() or torch.isnan(output).any() or torch.isnan(yhat).any():
            print(loss, torch.isnan(german).any(), torch.isnan(english).any(), torch.isnan(output).any(), torch.isnan(yhat).any())
            nan_counter += 1
            if nan_counter == 10:
                print("too many NaN's")
                break
            continue
        # write the loss
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()

        # update the weights
        optimizer.step()

        writer.flush()
        sum_loss += loss.item()
        count += 1
        loop.set_postfix_str(f"epoch: {epoch}/{config.epochs}  loss: {loss.item()}  avg loss: {sum_loss/count}")

    # Save the model and optimizer
    print(f"avg loss: {sum_loss/count}")
    path_to_save = "./checkpoints_1024/model_post_" + str(epoch)
    # save_checkpoint(model, optimizer, path_to_save, sum_loss/count, config)
    continue

    # Compute average loss on the mixed48k set
    model.eval()  # Set the model to evaluation mode turning off dropout
    val_loss = 0.0
    with torch.no_grad():  # No gradient computation for validation
        for batch in tqdm(test_dataloader):
            german_test = batch["output"].to(config.device).squeeze()
            english_test = batch["input"].to(config.device).squeeze()
            output_test = batch["output"].to(config.device).squeeze()
            yhat = model(german_test, english_test)
            loss = Fun.cross_entropy(
                yhat.view(-1, yhat.size(-1)), output_test.view(-1), ignore_index=0
            )
            val_loss += loss.item()
        avg_val_loss = val_loss / len(test_dataloader)
        writer.add_scalar("Loss/mixed48k", avg_val_loss, epoch)
        print(f"Epoch: {epoch},  Avg_val_loss: {avg_val_loss}")