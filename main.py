import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import *
from network import ViT

if __name__ == '__main__':
    train_dl = DataLoader(
        datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose(
            [
                transforms.ToTensor(), # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
            ]
            )
        ),
        batch_size=32,
        shuffle=True
    )

    val_dl = DataLoader(
        datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose(
            [
                transforms.ToTensor(), # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
            ]
            )
        ),
        batch_size=1024,
        shuffle=False
    )

    device = get_default_device()

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    model = to_device(ViT(), device)

    history = fit(
        epochs=100, 
        lr=0.001, 
        model=model, 
        train_loader=train_dl, 
        val_loader=val_dl
    )

    plot_losses(history)
    plot_accuracies(history)
