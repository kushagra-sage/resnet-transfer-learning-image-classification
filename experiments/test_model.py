import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet import get_resnet18

# Glacier dataset
model_glacier = get_resnet18(num_classes=6, pretrained=True)
print("Glacier model output classes:", model_glacier.fc.out_features)

# Brain tumor dataset
model_brain = get_resnet18(num_classes=2, pretrained=True)
print("Brain tumor model output classes:", model_brain.fc.out_features)
