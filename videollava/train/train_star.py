import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import sys, json, os
import numpy as np
import collections
import importlib
import argparse
import random
import wandb

@dataclass
class STaRArguments:
    number_cycles: int = 10
    star_config: str = "videostar/configs/star.yaml"
    star_path: str = "videollava_star"
    train_path: str = "videollava/train/train.py"
    train_args: str = "videollava_args.yaml"
    star_data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    aux_data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the auxiliary data."})
    

def generate_answers():
    pass

def label_rationalization():
    pass



def star_train():
    args = STaRArguments()
    train = importlib.import_module(args.train_path)
    for cycle in range(args.number_cycles):
        generate_answers()
        
        label_rationalization()
        
        train()
        

if __name__ == "__main__":
    star_train()
