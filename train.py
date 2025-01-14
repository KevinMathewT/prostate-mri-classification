import argparse

import yaml
import torch
from accelerate import Accelerator

from model import get_model
from loader.data import get_loaders
from criterion import get_criterion
from optimizer import get_optimizer, get_scheduler
from engine import train_one_epoch
from utils import setup_distributed, init_wandb, get_model_memory_footprint


def main(config):
    setup_distributed()

    accelerator = Accelerator()

    train_loader, valid_loaders = get_loaders(config)

    # Initialize model, criterion, optimizer, scheduler
    model = get_model(config)
    criterion = get_criterion(config)
    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config, train_loader)

    accelerator.print(f"----- Model -----")
    accelerator.print(model)
    accelerator.print(f"-----------------")

    accelerator.print(f"----- Criterion, Optimizer, Scheduler -----")
    accelerator.print(criterion)
    accelerator.print(optimizer)
    accelerator.print(scheduler)
    accelerator.print(f"-------------------------------------------")

    # Prepare everything with accelerator
    model, optimizer, scheduler, criterion, train_loader = accelerator.prepare(
        model, optimizer, scheduler, criterion, train_loader
    )

    # Prepare each validation loader individually
    valid_loaders = {
        key: accelerator.prepare(loader) for key, loader in valid_loaders.items()
    }

    accelerator.print(f"Global batch size: {config['data']['batch_size']}")
    accelerator.print(f"Local batch size per GPU: {len(next(iter(train_loader))['vol'])}")
    accelerator.print(f"Number of GPUs: {accelerator.num_processes}")
    accelerator.print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    accelerator.print(f"Model memory footprint: {get_model_memory_footprint(model):.2f}MB")
    accelerator.print(f"Using sampler: {train_loader.sampler}")
    print(torch.cuda.memory_summary(device=f"cuda:{accelerator.local_process_index}"))



    for epoch in range(config['train']['epochs']):
        train_one_epoch(model, train_loader, valid_loaders, criterion, optimizer, scheduler, accelerator, epoch, config)

# Command to run the training script
# resnet18, resnet50
# accelerate launch -m train config/resnet18.yaml
# accelerate launch -m train config/resnet50.yaml
# seresnext50
# accelerate launch -m train config/seresnext50.yaml
# accelerate launch -m train config/seresnext/seresnext_inc_bs_inc_epochs.yaml
# accelerate launch -m train config/seresnext/seresnext_dec_lr_inc_epochs_dropout_0.5.yaml
# accelerate launch -m train config/seresnext/seresnext_dec_lr_inc_epochs_dropout_0.5_with_aug.yaml
# accelerate launch -m train config/seresnext/seresnext_dec_lr_inc_epochs_dropout_0.5_with_aug_monai.yaml
# accelerate launch -m train config/seresnext/seresnext101_dec_lr_inc_epochs_dropout_0.5_with_aug_monai.yaml
# vit
# accelerate launch -m train config/vit.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with a given configuration file.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb
    init_wandb(config, args.config)

    main(config)
