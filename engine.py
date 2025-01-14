import torch
import torch.nn.functional as F

import wandb
from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import Binarizer

from utils import update_log_dict_with_binarization, save_best_model


def train_one_epoch(model, train_loader, valid_loaders, criterion, optimizer, scheduler, accelerator, epoch, config):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['vol']  # (B, 1, D, H, W)
        targets = batch['label']  # (B,)
        outputs = model(inputs)  # (B, num_classes)
        loss = criterion(outputs, targets)  # scalar

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()  # scalar
        _, predicted = outputs.max(1)  # (B,)
        probabilities = F.softmax(outputs, dim=1)  # (B, num_classes)

        total += targets.size(0)  # scalar
        correct += predicted.eq(targets).sum().item()  # scalar

        all_targets.extend(targets.cpu().numpy())  # (B,)
        all_predictions.extend(predicted.cpu().numpy())  # (B,)
        all_probabilities.extend(probabilities.detach().cpu().numpy())  # (B, num_classes)

        if accelerator.is_local_main_process:
            wandb.log({
                'epoch': epoch,
                'train_batch_idx': batch_idx,
                'train_batch_loss': loss.item(),
                'train_batch_accuracy': 100. * correct / total,
                'train_batch_learning_rate': scheduler.get_last_lr()[0]
            })

        if batch_idx == 0 or (batch_idx + 1) % 5 == 0 or batch_idx == len(train_loader) - 1:
            accelerator.print(f'[{epoch+1}/{config["train"]["epochs"]}][{batch_idx+1}/{len(train_loader)}] train loss: {loss.item():.10f} | lr: {scheduler.get_last_lr()[0]:.10f} | grad norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), 10000.0)}')

        k = 2
        if (batch_idx + 1) % (len(train_loader) // k) == 0:
            for valid_loader_id, valid_loader in valid_loaders.items():
                valid_one_epoch(valid_loader_id, model, valid_loader, criterion, accelerator, epoch, config)
            model.train()

    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)  # scalar
    recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)  # scalar
    # auc = roc_auc_score(all_targets, all_probabilities, average='macro', multi_class='ovr')  # scalar

    if config["train"]["binarize"]:  # Binary classification
        auc = roc_auc_score(all_targets, [p[1] for p in all_probabilities], average='macro')  # scalar
    else:  # Multiclass classification
        auc = roc_auc_score(all_targets, all_probabilities, average='macro', multi_class='ovr')  # scalar

    log_dict = {
        'epoch': epoch,
        'train_epoch_loss': total_loss / len(train_loader),  # scalar
        'train_epoch_accuracy': 100. * correct / total,  # scalar
        'train_epoch_precision': precision,  # scalar
        'train_epoch_recall': recall,  # scalar
        'train_epoch_auc': auc  # scalar
    }

    if not config["train"]["binarize"]:
        update_log_dict_with_binarization(log_dict, all_targets, all_predictions, all_probabilities, 'training')

    if accelerator.is_local_main_process:
        wandb.log(log_dict)

    accelerator.print(f'[{epoch+1}/{config["train"]["epochs"]}] train epoch loss: {total_loss / len(train_loader):.10f}')
    accelerator.print(f'[{epoch+1}/{config["train"]["epochs"]}] train accuracy: {100. * correct / total:.3f} | precision: {precision:.3f} | recall: {recall:.3f} | auc: {auc:.3f}')
    
    if not config["train"]["binarize"]:
        accelerator.print(f'[{epoch+1}/{config["train"]["epochs"]}] bin accuracy: {log_dict["training_bin_accuracy"]:.3f}% | bin precision: {log_dict["training_bin_precision"]:.3f} | bin recall: {log_dict["training_bin_recall"]:.3f} | bin auc: {log_dict["training_bin_auc"]:.3f}')


def valid_one_epoch(valid_loader_id, model, valid_loader, criterion, accelerator, epoch, config):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            inputs = batch['vol']  # (B, 1, D, H, W)
            targets = batch['label']  # (B,)
            outputs = model(inputs)  # (B, num_classes)
            loss = criterion(outputs, targets)  # scalar

            total_loss += loss.item()  # scalar
            _, predicted = outputs.max(1)  # (B,)
            probabilities = F.softmax(outputs, dim=1)  # (B, num_classes)

            total += targets.size(0)  # scalar
            correct += predicted.eq(targets).sum().item()  # scalar

            all_targets.extend(targets.cpu().numpy())  # (B,)
            all_predictions.extend(predicted.cpu().numpy())  # (B,)
            all_probabilities.extend(probabilities.detach().cpu().numpy())  # (B, num_classes)

            if accelerator.is_local_main_process:
                wandb.log({
                    'epoch': epoch,
                    f'{valid_loader_id}_batch_idx': batch_idx,
                    f'{valid_loader_id}_batch_loss': loss.item(),
                    f'{valid_loader_id}_batch_accuracy': 100. * correct / total,
                })

            if batch_idx == 0 or (batch_idx + 1) % 5 == 0 or batch_idx == len(valid_loader) - 1:
                accelerator.print(f'[{epoch+1}/{config["train"]["epochs"]}][{batch_idx+1}/{len(valid_loader)}] valid loss: {loss.item():.10f}')

    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)  # scalar
    recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)  # scalar
    # auc = roc_auc_score(all_targets, all_probabilities, average='macro', multi_class='ovr')  # scalar

    if config["train"]["binarize"]:  # Binary classification
        auc = roc_auc_score(all_targets, [p[1] for p in all_probabilities], average='macro')  # scalar
    else:  # Multiclass classification
        auc = roc_auc_score(all_targets, all_probabilities, average='macro', multi_class='ovr')  # scalar

    log_dict = {
        'epoch': epoch,
        f'{valid_loader_id}_epoch_loss': total_loss / len(valid_loader),  # scalar
        f'{valid_loader_id}_epoch_accuracy': 100. * correct / total,  # scalar
        f'{valid_loader_id}_epoch_precision': precision,  # scalar
        f'{valid_loader_id}_epoch_recall': recall,  # scalar
        f'{valid_loader_id}_epoch_auc': auc  # scalar
    }

    if not config["train"]["binarize"]:
        update_log_dict_with_binarization(log_dict, all_targets, all_predictions, all_probabilities, 'valid')

    if accelerator.is_local_main_process:
        wandb.log(log_dict)

    accelerator.print(f'[{epoch+1}/{config["train"]["epochs"]}] {valid_loader_id} epoch loss: {total_loss / len(valid_loader):.10f}')
    accelerator.print(f'[{epoch+1}/{config["train"]["epochs"]}] {valid_loader_id} accuracy: {100. * correct / total:.3f} | precision: {precision:.3f} | recall: {recall:.3f} | auc: {auc:.3f}')
    
    if not config["train"]["binarize"]:
        accelerator.print(f'[{epoch+1}/{config["train"]["epochs"]}] {valid_loader_id} bin accuracy: {log_dict["valid_bin_accuracy"]:.3f}% | bin precision: {log_dict["valid_bin_precision"]:.3f} | bin recall: {log_dict["valid_bin_recall"]:.3f} | bin auc: {log_dict["valid_bin_auc"]:.3f}')

    if accelerator.is_local_main_process:
        save_best_model(model, auc, 100. * correct / total, precision, recall, epoch, log_dict, valid_loader_id, accelerator=accelerator)
