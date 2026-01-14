import torch.nn as nn
from torchvision import models


def get_resnet(
    variant="resnet18",
    num_classes=2,
    pretrained=False,
    dropout_rate=0.0
):
    if variant == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
    elif variant == "resnet34":
        model = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )
    else:
        raise ValueError("Unsupported ResNet variant")

    in_features = model.fc.in_features

    # Custom classifier head (ARCHITECTURAL CHANGE)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    return model
