import torch
from cs336_systems.config import Systems_Config


def get_random_batch(config: Systems_Config, device):
    x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)
    y = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length), device=device)

    return x, y