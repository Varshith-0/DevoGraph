"""HNNs C Elegans Embryogenesis

Contributer: Lalith Bharadwaj Baru
"""
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
