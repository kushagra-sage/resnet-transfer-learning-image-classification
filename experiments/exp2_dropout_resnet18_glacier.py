import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------
# Add project root to Python path
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import get_dataloaders
from src.models.resnet import get_resnet
from src.training.train import train_one_epoch, evaluate

# -------------------------------------------------
# EXPERIMENT 2: DROPOUT VARIATION
# -------------------------------------------------
DATA_DIR = "data/intel_glacier"
NUM_CLASSES = 6

MODEL_VARIANT = "resnet18"
PRETRAINED = True
DROPOUT_RATE = 0.5     # 🔴 ONLY CHANGE FROM BASELINE

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
OPTIMIZER = "adam"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------------------------


def main():
    print("=== EXPERIMENT 2: DROPOUT COMPARISON ===")
    print(f"Device: {DEVICE}")
    print("Model: ResNet18 (Pretrained)")
    print("Optimizer: Adam")
    print("Dropout: 0.5")
    print("Batch Size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("-" * 40)

    # Load data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )

    # Load model (with dropout)
    model = get_resnet(
        variant=MODEL_VARIANT,
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        dropout_rate=DROPOUT_RATE
    )
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f}\n"
            f"Val   Acc: {val_acc:.4f} | Val   Loss: {val_loss:.4f}"
        )

    total_time = time.time() - start_time
    print("\nTraining completed")
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
