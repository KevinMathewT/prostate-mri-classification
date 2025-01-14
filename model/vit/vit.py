import torch
import torch.nn as nn
from monai.networks.nets import ViT

from utils import get_series_properties

class MRI_ViT_3D(nn.Module):
    def __init__(self, config):
        super(MRI_ViT_3D, self).__init__()
        self.config = config
        feature_dim = 1024
        num_heads = 16 # should divide feature_dim

        # Helper function to initialize ViT for a given series
        def init_vit(series_key, in_channels):
            props = get_series_properties(series_key)
            patch_size = {
                "axt2": (4, 16, 16),
                "adc": (2, 26, 26),
                "b1500": (2, 26, 26)
            }[series_key]
            return ViT(
                in_channels=in_channels,
                img_size=(props["num_slices"], props["crop_size"][0], props["crop_size"][1]),
                patch_size=patch_size,
                hidden_size=feature_dim,
                mlp_dim=3072,
                num_layers=6,
                num_heads=num_heads,
                spatial_dims=3,
                classification=False,
                save_attn=True,
                post_activation="Identity",
                dropout_rate=0.5,
            )

        # Define ViTs for different series
        self.axt2_vit = init_vit("axt2", in_channels=1)

        if config["train"]["model"]["combine_adc_b1500"]:
            self.adc_b1500_vit = init_vit("adc", in_channels=2)
        else:
            self.adc_vit = init_vit("adc", in_channels=1)
            self.b1500_vit = init_vit("b1500", in_channels=1)

        # Pooling mechanism
        self.pooling_type = config["train"]["model"]["vit"]["pooling"]

        # Fully connected classification head
        num_inputs = feature_dim * 2 if config["train"]["model"]["combine_adc_b1500"] else feature_dim * 3
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, config["train"]["model"]["num_classes"])
        )

    def pool_features(self, features, attention=None):
        if self.pooling_type == "mean":
            return features.mean(dim=1)  # Mean pooling
        elif self.pooling_type == "max":
            return features.max(dim=1).values  # Max pooling
        elif self.pooling_type == "cls":
            return features[:, 0, :]  # CLS token pooling
        elif self.pooling_type == "attention":
            attn_weights = attention.mean(dim=1)  # Average over heads
            attn_weights = attn_weights.softmax(dim=1)  # Normalize
            return (features * attn_weights.unsqueeze(-1)).sum(dim=1)  # Weighted sum
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def forward(self, x):
        axt2 = x['axt2']
        adc = x['adc']
        b1500 = x['b1500']

        # Extract features for AXT2
        axt2_features, axt2_attention = self.axt2_vit(axt2)
        axt2_features = self.pool_features(axt2_features, axt2_attention)

        # Combine ADC and B1500 features
        if self.config["train"]["model"]["combine_adc_b1500"]:
            adc_b1500 = torch.cat([adc, b1500], dim=1)
            adc_b1500_features, adc_b1500_attention = self.adc_b1500_vit(adc_b1500)
            adc_b1500_features = self.pool_features(adc_b1500_features, adc_b1500_attention)
            combined_features = torch.cat([axt2_features, adc_b1500_features], dim=1)
        else:
            adc_features, adc_attention = self.adc_vit(adc)
            adc_features = self.pool_features(adc_features, adc_attention)
            b1500_features, b1500_attention = self.b1500_vit(b1500)
            b1500_features = self.pool_features(b1500_features, b1500_attention)
            combined_features = torch.cat([axt2_features, adc_features, b1500_features], dim=1)

        # Classification layer
        output = self.fc(combined_features)
        return output
