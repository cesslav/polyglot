from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from v4_imp import Transformer, tokenize_fn
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC_LANG = "ru"
TGT_LANG = "en"
N_HEADS = 8
MODEL_DIM = N_HEADS * 32
NUM_LAYERS = 24
FF_DIM = MODEL_DIM * 4
DROPOUT = 0.1
BATCH_SIZE = 6
MAX_LEN = 512
NUM_EPOCHS = 50
LR = 1e-4
save_coef = 0.025


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare(rank, world_size, batch_size=32, pin_memory=True, num_workers=0):
    dataset = load_dataset("wmt/wmt19", f"{SRC_LANG}-{TGT_LANG}", split="train")
    dataset.set_format(type='torch')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=True, sampler=sampler)

    return dataloader


def main(rank, transformer, ):
    # setup the process groups
    setup(rank, world_size)  # prepare the dataloader
    dataloader = prepare(rank, world_size)

    # instantiate the model(it's your own model) and move it to the right device
    model = transformer().to(rank)

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)