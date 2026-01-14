import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes, pretrained=False, freeze_backbone=False):
    """
    Returns a ResNet18 model.
    
    Args:
        num_classes (int): number of output classes
        pretrained (bool): use ImageNet pretrained weights or not
        freeze_backbone (bool): freeze convolutional layers
    """

    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    # Freeze backbone if required
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final classification layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
