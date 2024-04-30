import os
from datetime import timedelta
import torch
import torch.distributed as dist
from cs336_basics.configs.config import Config
from cs336_basics.model.transformer import Transformer
from cs336_basics.training.optimizer import AdamW
from cs336_basics.model.util import crossEntropyLoss, load_model, save_model, get_batch, log_validation_loss, log, build_model
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

class DDP_Individual_Parameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, backend="nccl") -> None:
        super(DDP_Individual_Parameters, self).__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.module = module
        self.backend = backend

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        self.handles = []
    
    def forward(self, *inputs, **kwargs):

        y = self.module(*inputs, **kwargs)
        return y

    def finish_gradient_synchronization(self):
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                handle = dist.all_reduce(param.grad.data, async_op=True)
                self.handles.append(handle)

        for handle in self.handles:
            handle.wait()

        self.handles.clear()