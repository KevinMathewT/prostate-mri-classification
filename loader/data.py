import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from .preprocessing import preprocess_volume, get_series_properties
from concurrent.futures import ThreadPoolExecutor


class ExamH5(Dataset):
    def __init__(self, config, df_meta_path):
        super().__init__()
        self.config = config
        self.series = config["data"]["series"]
        self.datadir = config["paths"]["data_dir"]
        self.df_metadata = pd.read_csv(df_meta_path)

        # preprocessing options
        self.augment = config["data"]["augment_data"]
        self.normalize = config["data"]["normalize"]
        self.zero_pad = config["data"]["zero_pad"]

        self.class_weights = self.calculate_class_weights(config, df_meta_path)

    def calculate_class_weights(self, config, csv_path):
        num_classes = config["train"]["model"]["num_classes"]
        df = pd.read_csv(csv_path)
        if config["train"]["binarize"]:
            labels = (df["maxPIRADS"] > 2).astype(int)  # Binarize: 1 and 2 -> 0, >2 -> 1
        else:
            labels = df["maxPIRADS"].astype(int) - 1  # Adjust labels to be 0-4
        label_counts = labels.value_counts().sort_index()
        total_samples = len(labels)
        class_weights = total_samples / (num_classes * label_counts)
        return torch.tensor(class_weights.values, dtype=torch.float)

    def __getitem__(self, index):
        accession_number = int(self.df_metadata.iloc[index]["AccessionNumber"])
        patient_id = int(self.df_metadata.iloc[index]["PatientID"])
        path = Path(self.datadir) / f"{accession_number}.h5"
        feature_dict = {}

        with h5py.File(path, "r") as f:
            max_pirads = int(f.attrs["maxPIRADS"])
            if self.config["train"]["binarize"]:
                label = 1 if max_pirads > 2 else 0  # Binarize: 1 and 2 -> 0, >2 -> 1
            else:
                label = max_pirads - 1  # Adjust labels to be 0-4

            for s in self.series:
                if s not in f:
                    raise ValueError(f"Series {s} not found in file {path}")

                vol = f[s][:]  # Load the volume as a NumPy array
                series_props = get_series_properties(s)
                vol = preprocess_volume(
                    vol,
                    series_key=s,
                    normalize=self.normalize,
                    augment=self.augment,
                    slice_by_slice=False,
                    zero_pad=self.zero_pad,
                    config=self.config,
                )  # Preprocess the volume
                vol = torch.FloatTensor(vol).unsqueeze(0)  # Convert to a PyTorch tensor
                feature_dict[s] = vol

        return {
            "vol": feature_dict,
            "label": label,
            "maxPIRADS": max_pirads,
            "AccessionNumber": accession_number,
            "PatientID": patient_id,
            "location": str(path),
        }

    def __len__(self):
        return self.df_metadata.shape[0]


def get_loaders(config):
    train_dataset = ExamH5(config, config["paths"]["train_csv"])
    valid_csvs = config["paths"]["valid_csv"]

    config["data"]["class_weights"] = train_dataset.class_weights

    is_distributed = torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() if is_distributed else 1

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"] // world_size,  # Scale batch size for distributed
        shuffle=(train_sampler is None),  # Shuffle only if not using sampler
        num_workers=config["data"]["num_workers"],
        sampler=train_sampler,
    )

    valid_loaders = {}
    for i, valid_csv in enumerate(valid_csvs):
        valid_dataset = ExamH5(config, valid_csv)
        valid_sampler = DistributedSampler(valid_dataset) if is_distributed else None
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config["data"]["batch_size"] // world_size,
            shuffle=False,  # Validation data should not be shuffled
            num_workers=config["data"]["num_workers"],
            sampler=valid_sampler,
        )
        valid_loaders[f"valid_loader_{i}"] = valid_loader

    # Debug print after loaders are created
    print(f"Global batch size: {config['data']['batch_size']}")
    if train_sampler:
        print(f"Using DistributedSampler for train_loader")
    else:
        print(f"Using SequentialSampler for train_loader")

    for batch in train_loader:
        series_keys = list(batch["vol"].keys())  # Dynamically check available keys
        print(f"Local batch size per GPU: {batch['vol'][series_keys[0]].size(0)}")
        break

    return train_loader, valid_loaders

# main here is just for testing purposes
def main():
    config = {
        "data": {
            "series": ["axt2", "adc", "b1500"],
            "val_split": 0.2,
            "batch_size": 4,
            "num_workers": 2,
            "augment_data": False,
            "normalize": True,
            "zero_pad": True,
        },
        "paths": {
            "data_dir": "/gpfs/data/prostatelab/processed_data/h5/t2_dwi_filtered_20240212",
            "train_csv": "/gpfs/data/prostatelab/processed_data/csv/data_split/train_temporal_split_20240619_exclude_mismatch.csv",
        },
    }

    train_loader, val_loader = get_loaders(config)

    # Get the first batch from the training loader
    first_batch = next(iter(train_loader))

    # Print the contents and dimensionality of the first batch
    for key, value in first_batch.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value.shape}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
