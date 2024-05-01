import os
from datetime import timedelta
import torch
import torch.distributed as dist
from cs336_basics.configs.config import Config
from cs336_basics.model.transformer import Transformer
from cs336_basics.training.optimizer import AdamW
from cs336_basics.model.util import crossEntropyLoss, load_model, save_model, get_batch, log_validation_loss, log, build_model
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy

class DDP_Individual_Parameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super(DDP_Individual_Parameters, self).__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.handles = []

        def register_post_accumulate_grad_hook(param):
            grad = param.grad
            if grad is not None:
                grad /= self.world_size
                handle = dist.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)
                

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(register_post_accumulate_grad_hook)
            dist.broadcast(param.data, src=0)
    
    def forward(self, *inputs, **kwargs):
        y = self.module(*inputs, **kwargs)
        return y

    def finish_gradient_synchronization(self):

        for handle in self.handles:
            handle.wait()
        self.handles.clear()

class DDP_Bucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float) -> None:
        super(DDP_Bucketed, self).__init__()

        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized")

        self.module = module

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.bucket_size_mb = bucket_size_mb

        self.handles = []
        self.buckets = []
        

        def register_post_accumulate_grad_hook(param, bucket_index):
            self.bucket_ready[bucket_index] += 1
            if self.bucket_ready[bucket_index] == len(self.buckets[bucket_index]):
                grads = [param.grad for param in self.buckets[bucket_index]]
                if grads:
                    dense = _flatten_dense_tensors(grads)
                    handle = dist.all_reduce(dense, op=torch.distributed.ReduceOp.SUM, async_op=True)
                    self.handles.append((handle, dense, bucket_index))
                self.bucket_ready[bucket_index] = 0
                

        current_bucket = []
        current_bucket_size = 0
        
        mb_to_bytes = 1024**2
        max_bucket_bytes = self.bucket_size_mb * mb_to_bytes

        for param in reversed(list(self.module.parameters())):
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param_size = param.data.numel() * param.data.element_size()
                if current_bucket_size + param_size > max_bucket_bytes:
                    self.buckets.append(current_bucket)
                    current_bucket = []
                    current_bucket_size = 0
                current_bucket.append(param)
                current_bucket_size += param_size
        
        if current_bucket:
            self.buckets.append(current_bucket)

        self.bucket_ready = [0 for _ in range(len(self.buckets))]

        for i, bucket in enumerate(self.buckets):
            for param in bucket:
                param.register_post_accumulate_grad_hook(lambda param, bucket_index=i: register_post_accumulate_grad_hook(param, bucket_index))
    
    def forward(self, *inputs, **kwargs):
        y = self.module(*inputs, **kwargs)
        return y

    def finish_gradient_synchronization(self):

        for handle, dense, bucket_index in self.handles:
            grads = [param.grad for param in self.buckets[bucket_index] if param.grad is not None]
            handle.wait()
            unflattened_tensors = _unflatten_dense_tensors(dense, grads)
            for param, unflattened in zip(self.buckets[bucket_index], unflattened_tensors):
                param.grad.data = unflattened / self.world_size
        self.handles.clear()
