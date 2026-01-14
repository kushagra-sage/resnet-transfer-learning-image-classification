import sys, os, time
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.dataloader import get_dataloaders
from src.models.resnet import get_resnet
from src.training.train import train_one_epoch, evaluate

# ---------------- CONFIG ----------------
DATA_DIR = "data/brain_tumor"
NUM_CLASSES = 2

MODEL_VARIANT = "resnet18"
PRETRAINED = True
DROPOUT_RATE = 0.0

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
OPTIMIZER = "adam"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------


def main():
    print("=== BASELINE | BRAIN TUMOR ===")

    train_loader, val_loader, _ = get_dataloaders(
        DATA_DIR, BATCH_SIZE
    )

    model = get_resnet(
        MODEL_VARIANT, NUM_CLASSES, PRETRAINED, DROPOUT_RATE
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )
        print(f"Epoch {epoch+1}: Val Acc={val_acc:.4f}")

    print("Time:", time.time() - start)


if __name__ == "__main__":
    main()
