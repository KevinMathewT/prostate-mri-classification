import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import torch.nn.functional as F


class OrdinalCrossEntropyLoss:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, logits, targets):
        thresholds = torch.arange(1, self.num_classes, device=logits.device).view(1, -1)
        targets_expanded = targets.unsqueeze(1)
        binary_targets = (targets_expanded > thresholds).float()
        log_probs = F.logsigmoid(logits)
        loss = -(
            binary_targets * log_probs + (1 - binary_targets) * F.logsigmoid(-logits)
        ).mean()
        return loss


class EarthMoversDistanceLoss:
    def __call__(self, logits, targets):
        cumulative_preds = torch.cumsum(F.softmax(logits, dim=1), dim=1)
        cumulative_targets = torch.cumsum(
            F.one_hot(targets, num_classes=logits.size(1)).float(), dim=1
        )
        return torch.mean(
            torch.sum(torch.abs(cumulative_preds - cumulative_targets), dim=1)
        )


def get_criterion(config):
    criterion_name = config["train"]["criterion"]
    num_classes = config["train"]["model"]["num_classes"]
    weights = config["data"].get("class_weights", None)
    if weights is not None:
        print(f"Using weights: {weights}")
        weights = torch.tensor(weights, dtype=torch.float32)
    else:
        print("No weights specified")

    if criterion_name == "ce":
        return CrossEntropyLoss()
    elif criterion_name == "ce_weighted":
        if weights is None:
            raise ValueError("weights must be specified for ce_weighted.")
        return CrossEntropyLoss(weight=weights)
    elif criterion_name == "bce":
        return BCELoss()
    elif criterion_name == "bce_weighted":
        if weights is None:
            raise ValueError("weights must be specified for bce_weighted.")
        return BCELoss(weight=weights)
    elif criterion_name == "mse":
        return MSELoss()
    elif criterion_name == "mse_weighted":
        if weights is None:
            raise ValueError("weights must be specified for mse_weighted.")
        # PyTorch does not have built-in support for weighted MSE
        def weighted_mse_loss(predictions, targets):
            loss = F.mse_loss(predictions, targets, reduction="none")
            return torch.mean(loss * weights)
        return weighted_mse_loss
    elif criterion_name == "ordinal_ce":
        if num_classes is None:
            raise ValueError("num_classes must be specified for ordinal_ce.")
        return OrdinalCrossEntropyLoss(num_classes)
    elif criterion_name == "emd":
        return EarthMoversDistanceLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")
