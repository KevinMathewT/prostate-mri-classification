import math

import torch
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR

def get_optimizer(parameters, config):
    optim_name = config["train"]["optimizer"]["name"]
    lr = config["train"]["optimizer"]["lr"]

    if optim_name == "adam":
        return Adam(parameters, lr=lr, weight_decay=0.01)
    if optim_name == "adamw":
        return Adam(parameters, lr=lr, weight_decay=0.001)
    elif optim_name == "sgd":
        return SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.01)
    elif optim_name == "rmsprop":
        return RMSprop(parameters, lr=lr, momentum=0.9, weight_decay=0.01)
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

def get_scheduler(optimizer, config, train_loader):
    sched_name = config["train"]["scheduler"]["name"]
    epochs = config["train"]["epochs"]
    total_steps = len(train_loader) * epochs

    if sched_name == "steplr":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    elif sched_name == "exponentiallr":
        return ExponentialLR(optimizer, gamma=0.95)
    elif sched_name == "reducelronplateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    elif sched_name == "cosine":
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            cosine_progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1 + math.cos(math.pi * cosine_progress)))
        return LambdaLR(optimizer, lr_lambda)
    elif sched_name == "onecycle":
        return OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=epochs)
    elif sched_name == "linear":
        def lr_lambda(current_step):
            warmup_steps = 0.1 * epochs * len(train_loader)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(epochs - current_step) / float(max(1, epochs - warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")
