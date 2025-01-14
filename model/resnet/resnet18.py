import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r3d_18


def r3d_18_single_channel_input(in_channels, pretrained=False, progress=True, **kwargs):
    '''
    modified r3d_18 model to accept single-channel input. 
    The first convolutional layer is modified to accept single-channel input. 
    The weights of the first convolutional layer are initialized based on the mean of the original weights. 
    The rest of the model is the same as the original r3d_18 model. 
    '''
    # Load the original r3d_18 model
    model = r3d_18(pretrained=pretrained, progress=progress, **kwargs)

    # Modify the convolutional layer in BasicStem to accept single-channel input
    original_conv = model.stem[0]
    new_conv = nn.Conv3d(
        in_channels=in_channels,  # Change input channels to 1
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    # Optionally initialize the new convolutional weights using the mean of pretrained weights
    if pretrained:
        with torch.no_grad():
            new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))

    # Replace only the first convolutional layer in the stem
    model.stem[0] = new_conv

    return model


class MRI_ResNet18_3D(nn.Module):
    def __init__(self, config):
        super(MRI_ResNet18_3D, self).__init__()
        self.config = config

        # AXT2 network (always separate)
        self.axt2_resnet = r3d_18_single_channel_input(in_channels=1, pretrained=config["train"]["model"]["pretrained"])
        self.axt2_resnet.fc = nn.Identity()

        # ADC and B1500 network configuration
        if config["train"]["model"]["combine_adc_b1500"]:
            self.adc_b1500_resnet = r3d_18_single_channel_input(in_channels=2, pretrained=config["train"]["model"]["pretrained"])
            self.adc_b1500_resnet.fc = nn.Identity()
        else:
            self.adc_resnet = r3d_18_single_channel_input(in_channels=1, pretrained=config["train"]["model"]["pretrained"])
            self.adc_resnet.fc = nn.Identity()

            self.b1500_resnet = r3d_18_single_channel_input(in_channels=1, pretrained=config["train"]["model"]["pretrained"])
            self.b1500_resnet.fc = nn.Identity()

        # Fully connected layer for classification
        feature_dim = 512  # ResNet50 output size
        num_inputs = feature_dim * 2 if config["train"]["model"]["combine_adc_b1500"] else feature_dim * 3
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config["train"]["model"]["num_classes"])
        )

    def forward(self, x):
        axt2 = x['axt2']  # I: (B, 1, D, H, W)
        adc = x['adc']  # I: (B, 1, D, H, W)
        b1500 = x['b1500']  # I: (B, 1, D, H, W)

        axt2_features = self.axt2_resnet(axt2)  # I: (B, 1, D, H, W) | O: (B, 512)
        axt2_features = axt2_features.view(axt2_features.size(0), -1)  # I: (B, 512) | O: (B, 512)

        if self.config["train"]["model"]["combine_adc_b1500"]:
            adc_b1500 = torch.cat([adc, b1500], dim=1)  # I: (B, 1, D, H, W), (B, 1, D, H, W) | O: (B, 2, D, H, W)
            adc_b1500_features = self.adc_b1500_resnet(adc_b1500)  # I: (B, 2, D, H, W) | O: (B, 512)
            combined_features = torch.cat([axt2_features, adc_b1500_features], dim=1)  # I: (B, 512), (B, 512) | O: (B, 1024)
        else:
            adc_features = self.adc_resnet(adc)  # I: (B, 1, D, H, W) | O: (B, 512)
            b1500_features = self.b1500_resnet(b1500)  # I: (B, 1, D, H, W) | O: (B, 512)
            combined_features = torch.cat([axt2_features, adc_features, b1500_features], dim=1)  # I: (B, 512), (B, 512), (B, 512) | O: (B, 1536)

        output = self.fc(combined_features)  # I: (B, 1024) or (B, 1536) | O: (B, num_classes)
        return output