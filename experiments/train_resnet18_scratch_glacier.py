import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import get_dataloaders
from src.models.resnet import get_resnet18
from src.training.train import (
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint
)

# ===================== CONFIG =====================
DEBUG_MODE = True   # True = fast CPU | False = full training

DATA_DIR = "data/intel_glacier"
NUM_CLASSES = 6
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "models/resnet18_glacier_scratch.pth"

if DEBUG_MODE:
    EPOCHS = 1
    BATCH_SIZE = 16
else:
    EPOCHS = 10
    BATCH_SIZE = 32
# =================================================


def main():
    print("Device:", DEVICE)
    print("Debug mode:", DEBUG_MODE)

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )

    #  SCRATCH MODEL (NO PRETRAINING)
    model = get_resnet18(
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False
    )
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_acc = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print("Resuming training from checkpoint...")
        model, optimizer, start_epoch, best_acc = load_checkpoint(
            model, optimizer, CHECKPOINT_PATH, DEVICE
        )

    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}"
        )

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": max(best_acc, val_acc),
            },
            CHECKPOINT_PATH
        )

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
