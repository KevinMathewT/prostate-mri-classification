import os
import wandb
from accelerate import Accelerator

import torch
from torch.distributed import init_process_group, is_initialized

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import random

def seed_everything(seed):
    # Determine rank (default to 0 for non-DDP scripts)
    rank = int(os.environ.get("RANK", 0)) if "RANK" in os.environ else 0
    global_seed = seed + rank  # Adjust seed per process for DDP
    
    os.environ["PYTHONHASHSEED"] = str(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_files():
    extension_depth_pairs = [('.py', 3), ('.yaml', 2)]
    collected_files = []

    base_dir = os.path.abspath(".")
    base_depth = base_dir.count(os.sep)

    for extension, max_depth in extension_depth_pairs:
        for root, dirs, files in os.walk("."):
            # Get absolute path of the current directory
            root_abs = os.path.abspath(root)
            current_depth = root_abs.count(os.sep) - base_depth

            # Skip directories that exceed the max depth
            if current_depth >= max_depth:
                dirs[:] = []  # Prune traversal by clearing 'dirs'
                continue  # Skip processing files in this directory

            # Collect files matching the extension
            for file in files:
                if file.endswith(extension):
                    collected_files.append(os.path.relpath(os.path.join(root, file)))

    return collected_files

def get_model_memory_footprint(model, device='cuda'):
    model = model.to(device)
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())  # Parameters
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())    # Buffers
    total_memory = param_memory + buffer_memory
    memory_in_mb = total_memory / (1024 ** 2)  # Convert bytes to MB

    return memory_in_mb
        

def init_wandb(config, config_file):
    accelerator = Accelerator()
    if accelerator.is_local_main_process:  # Initialize wandb only on the main process
        run_name = os.path.basename(config_file).split('.')[0]
        wandb.init(project="Prostate MRI Classification", config=config, name=run_name)
        for f in log_files():
            print(f"Logging file: {f}")
            wandb.save(f)
    else:
        # Prevent other processes from logging to wandb
        os.environ["WANDB_MODE"] = "disabled"

def setup_distributed(seed=42):
    if torch.cuda.device_count() > 1 and not is_initialized():
        try:
            init_process_group(backend="nccl", init_method="env://")
        except ValueError as e:
            print(f"Distributed setup skipped: {e}")

    print(f"[{os.environ.get('RANK', 0)}] Setting random seed to {seed}")
    seed_everything(seed)


def binarize_labels(labels, threshold=2):
    return (labels >= threshold).astype(int)

def update_log_dict_with_binarization(log_dict, all_targets, all_predictions, all_probabilities, prefix):
    bin_targets = binarize_labels(np.array(all_targets))
    bin_predictions = binarize_labels(np.array(all_predictions))
    bin_auc = roc_auc_score(bin_targets, np.array(all_probabilities)[:, 1])

    bin_precision = precision_score(bin_targets, bin_predictions, average='macro', zero_division=0)
    bin_recall = recall_score(bin_targets, bin_predictions, average='macro', zero_division=0)
    bin_accuracy = (bin_targets == bin_predictions).mean()

    log_dict.update({
        f'{prefix}_bin_accuracy': bin_accuracy * 100,
        f'{prefix}_bin_precision': bin_precision,
        f'{prefix}_bin_recall': bin_recall,
        f'{prefix}_bin_auc': bin_auc
    })


def get_series_properties(series_key):
    """
    Get properties for a given series.

    Args:
        series_key (str): The series key, e.g., "axt2", "adc", or "b1500".

    Returns:
        dict: A dictionary containing crop_size, pixel_range, and num_slices.
    """
    series_properties = {
        "axt2": {"crop_size": (180, 180), "pixel_range": (0, 1), "num_slices": 30},
        "adc": {"crop_size": (130, 130), "pixel_range": (0, 1), "num_slices": 22},
        "b1500": {"crop_size": (130, 130), "pixel_range": (0, 1), "num_slices": 22},
    }

    if series_key not in series_properties:
        raise ValueError(f"Unknown series key: {series_key}")
    return series_properties[series_key]



# Global dictionary to track the best AUC for each validation loader
best_auc_dict = {}

# Function to save the best model based on AUC for a specific validation loader
def save_best_model(model, auc, accuracy, precision, recall, epoch, log_dict, valid_loader_id, save_dir="weights/", prefix="best_model", accelerator=None):
    """
    Saves the model if it has the best AUC so far for the given validation loader,
    deletes the previous best model for that loader.

    Args:
        model (torch.nn.Module): The model to be saved.
        auc (float): Current AUC to compare with the best.
        accuracy (float): Current accuracy to include in the filename.
        precision (float): Current precision to include in the filename.
        recall (float): Current recall to include in the filename.
        epoch (int): Current epoch.
        log_dict (dict): Dictionary containing metrics to include in the filename.
        valid_loader_id (str): Identifier for the validation loader.
        save_dir (str): Directory to save the model weights.
        prefix (str): Prefix for the model filename.
        accelerator (Accelerator, optional): Accelerator object for saving the model.
    """
    global best_auc_dict

    if valid_loader_id not in best_auc_dict or auc > best_auc_dict[valid_loader_id]:
        best_auc_dict[valid_loader_id] = auc

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_name = f"{prefix}_{valid_loader_id}_auc_{auc:.4f}_acc_{accuracy:.3f}_prec_{precision:.3f}_rec_{recall:.3f}_epoch_{epoch}.pt"
        model_path = os.path.join(save_dir, model_name)

        # Remove previous best model for this validation loader
        for filename in os.listdir(save_dir):
            if filename.startswith(f"{prefix}_{valid_loader_id}_"):
                os.remove(os.path.join(save_dir, filename))

        # Save using the accelerator
        if accelerator:
            accelerator.save({
                'model_state_dict': model.state_dict(),
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'epoch': epoch,
                'log_dict': log_dict
            }, model_path)
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'epoch': epoch,
                'log_dict': log_dict
            }, model_path)

        # Save to WandB
        wandb.save(model_path)

        print(f"New best model saved for {valid_loader_id}: {model_path}")
    else:
        print(f"AUC for {valid_loader_id} did not improve. Current: {auc:.4f}, Best: {best_auc_dict[valid_loader_id]:.4f}")
