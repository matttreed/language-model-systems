import torch
from typing import Type, Any
import torch.distributed as dist
from copy import deepcopy, copy

class Sharded_Optimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.all_params = list(params)
        self.sharded_params = [param for i, param in enumerate(self.all_params) if i % self.world_size == self.rank]
        self.optimizer = optimizer_cls(params=self.sharded_params, **kwargs)
        super().__init__(params=self.all_params, defaults=kwargs)

    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure=closure, **kwargs)
        for i, param in enumerate(self.all_params):
            dist.broadcast(param.data, src=i % self.world_size)

    def add_param_group(self, param_group: dict[str, Any]):
        # self.optimizer.add_param_group(param_group)
        super().add_param_group(param_group)




    # def _assign_params(self):
    #     for group in self.param_groups:
    #         for param in group['params']:
    #             dist.broadcast(param.data, src=0)

    # def _synchronize_params(self):
    #     for group in self.param_groups:
    #         for param in group['params']:
    #             dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    #             param.data /= dist.get_world_size()