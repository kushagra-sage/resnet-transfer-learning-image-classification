from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    data_dir,
    batch_size=32,
    img_size=224,
    num_workers=2
):
    """
    Generic dataloader for image classification datasets
    compatible with PyTorch ImageFolder structure.
    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, train_dataset.classes
