import torch
import torch.nn as nn
from monai.networks.nets import SEResNext101

class MRI_SEResNext101_3D(nn.Module):
    def __init__(self, config):
        super(MRI_SEResNext101_3D, self).__init__()
        self.config = config

        # AXT2 network
        self.axt2_seresnext = SEResNext101(
            spatial_dims=3,
            in_channels=1,  # Single-channel input for AXT2
            num_classes=0,  # Set to None to use as feature extractor
            pretrained=False,  # Pretraining not supported for 3D
            dropout_prob=config["train"]["model"]["dropout_prob"],
        )
        self.axt2_seresnext.last_linear = nn.Identity()  # Remove classification layer

        # ADC and B1500 network configuration
        if config["train"]["model"]["combine_adc_b1500"]:
            self.adc_b1500_seresnext = SEResNext101(
                spatial_dims=3,
                in_channels=2,  # Combine ADC and B1500 as channels
                num_classes=0,
                pretrained=False,
                dropout_prob=config["train"]["model"]["dropout_prob"],
            )
            self.adc_b1500_seresnext.last_linear = nn.Identity()
        else:
            self.adc_seresnext = SEResNext101(
                spatial_dims=3,
                in_channels=1,
                num_classes=0,
                pretrained=False,
                dropout_prob=config["train"]["model"]["dropout_prob"],
            )
            self.adc_seresnext.last_linear = nn.Identity()

            self.b1500_seresnext = SEResNext101(
                spatial_dims=3,
                in_channels=1,
                num_classes=0,
                pretrained=False
            )
            self.b1500_seresnext.last_linear = nn.Identity()

        # Fully connected layer for classification
        feature_dim = 2048  # SEResNext101 output size
        num_inputs = feature_dim * 2 if config["train"]["model"]["combine_adc_b1500"] else feature_dim * 3
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, config["train"]["model"]["num_classes"])
        )

    def forward(self, x):
        axt2 = x['axt2']
        adc = x['adc']
        b1500 = x['b1500']

        # Extract features for AXT2
        axt2_features = self.axt2_seresnext(axt2)
        axt2_features = axt2_features.view(axt2_features.size(0), -1)

        # Combine ADC and B1500 features
        if self.config["train"]["model"]["combine_adc_b1500"]:
            adc_b1500 = torch.cat([adc, b1500], dim=1)  # Combine as channels
            adc_b1500_features = self.adc_b1500_seresnext(adc_b1500)
            combined_features = torch.cat([axt2_features, adc_b1500_features], dim=1)
        else:
            adc_features = self.adc_seresnext(adc)
            b1500_features = self.b1500_seresnext(b1500)
            combined_features = torch.cat([axt2_features, adc_features, b1500_features], dim=1)

        # Classification layer
        output = self.fc(combined_features)
        return output
