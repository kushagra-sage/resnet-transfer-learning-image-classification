import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import get_dataloaders

# --------- Intel Glacier Dataset ----------
intel_data_dir = "data/intel_glacier"

intel_train_loader, intel_val_loader, intel_classes = get_dataloaders(
    data_dir=intel_data_dir,
    batch_size=32
)

print("Intel Glacier Dataset")
print("Classes:", intel_classes)
print("Number of classes:", len(intel_classes))
print("Training samples:", len(intel_train_loader.dataset))
print("Validation samples:", len(intel_val_loader.dataset))
print("-" * 40)


# --------- Brain Tumor Dataset ----------
brain_data_dir = "data/brain_tumor"

brain_train_loader, brain_val_loader, brain_classes = get_dataloaders(
    data_dir=brain_data_dir,
    batch_size=32
)

print("Brain Tumor Dataset")
print("Classes:", brain_classes)
print("Number of classes:", len(brain_classes))
print("Training samples:", len(brain_train_loader.dataset))
print("Validation samples:", len(brain_val_loader.dataset))
print("-" * 40)
