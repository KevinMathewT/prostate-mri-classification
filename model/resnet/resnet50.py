import torch
import torch.nn as nn
from monai.networks.nets.resnet import ResNet, ResNetBottleneck, get_inplanes

class MRI_ResNet50_3D(nn.Module):
    def __init__(self, config):
        super(MRI_ResNet50_3D, self).__init__()
        self.config = config

        # Common ResNet50 parameters
        block = ResNetBottleneck
        layers = [3, 4, 6, 3]  # ResNet50 configuration
        block_inplanes = get_inplanes()

        # AXT2 network
        self.axt2_resnet = ResNet(
            spatial_dims=3,
            n_input_channels=1,  # Single-channel input for AXT2
            block=block,
            layers=layers,
            block_inplanes=block_inplanes,
            num_classes=0  # Set to 0 to use as feature extractor
        )
        self.axt2_resnet.fc = nn.Identity()  # Remove classification layer

        # ADC and B1500 network configuration
        if config["train"]["model"]["combine_adc_b1500"]:
            self.adc_b1500_resnet = ResNet(
                spatial_dims=3,
                n_input_channels=2,  # Combine ADC and B1500 as channels
                block=block,
                layers=layers,
                block_inplanes=block_inplanes,
                num_classes=0
            )
            self.adc_b1500_resnet.fc = nn.Identity()
        else:
            self.adc_resnet = ResNet(
                spatial_dims=3,
                n_input_channels=1,
                block=block,
                layers=layers,
                block_inplanes=block_inplanes,
                num_classes=0
            )
            self.adc_resnet.fc = nn.Identity()

            self.b1500_resnet = ResNet(
                spatial_dims=3,
                n_input_channels=1,
                block=block,
                layers=layers,
                block_inplanes=block_inplanes,
                num_classes=0
            )
            self.b1500_resnet.fc = nn.Identity()

        # Fully connected layer for classification
        feature_dim = 2048  # ResNet50 output size
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
        axt2_features = self.axt2_resnet(axt2)
        axt2_features = axt2_features.view(axt2_features.size(0), -1)

        # Combine ADC and B1500 features
        if self.config["train"]["model"]["combine_adc_b1500"]:
            adc_b1500 = torch.cat([adc, b1500], dim=1)  # Combine as channels
            adc_b1500_features = self.adc_b1500_resnet(adc_b1500)
            combined_features = torch.cat([axt2_features, adc_b1500_features], dim=1)
        else:
            adc_features = self.adc_resnet(adc)
            b1500_features = self.b1500_resnet(b1500)
            combined_features = torch.cat([axt2_features, adc_features, b1500_features], dim=1)

        # Classification layer
        output = self.fc(combined_features)
        return output
