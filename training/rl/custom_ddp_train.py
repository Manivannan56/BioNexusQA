"""
custom_ddp_train.py
Production-grade, config-driven multi-GPU trainer using PyTorch DDP.
Features:
- Deterministic seeding + bf16 autocast
- Gradient accumulation + clipping
- LoRA (optional) with PEFT
- Periodic eval (loss + perplexity) and checkpointing
- Robust resume-from-checkpoint
- Rank-aware logging (only rank 0 prints/saves)
"""

import os,math,yaml,time,random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from transformers import(
    AutoTokenizer,AutoModelForCausalLM, get_scheduler
)
from peft import LoraConfig, get_peft_model
from rl.dataset_qa import

#-------------------Utils---------------

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.deterministic=True; cudnn.benchmark=False



#------------------------Main-------------

def main():
    cfg=yaml.safe_load(open("configs/sft_config.yaml"))
    os.make