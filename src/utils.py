from matplotlib import pyplot as plt

import os
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from torchvision import transforms as tt
from torchvision.io import read_image


def plot_losses(history, name='./losses'):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    
    plt.figure()
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(name + '.png')

def plot_accuracies(history, name='./accuracies'):
    accuracies = [x['val_acc'] for x in history]

    plt.figure()
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(name + '.png')

@torch.no_grad()
def plot_regression(model, test_dl, name='./regression'):
    targets=[]
    preds=[]

    for batch in test_dl:
        images, labels = batch
        predictions = model(images.float())
        targets += labels.tolist() * 100
        preds += predictions.tolist() * 100

    plt.figure()
    plt.scatter(targets, preds, s=2, alpha=0.5)
    plt.savefig(name + '.png')

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)

    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, grad_clip=0.1, steps_per_epoch=len(train_loader)):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []

        lr = []

        for batch in tdqm(train_loader, desc = 'Epoch #' + str(epoch) + '/' + str(epochs)):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lr.append(get_lr(optimizer))
            scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        if epoch % 20 == 0:
            plot_regression(model, val_loader, name="./" + str(num) + "_regression_" + str(epoch))

    return history

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def load_mnist(device=get_default_device):
    train_dl = DataLoader(
        datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=tt.Compose([
                tt.ToTensor(), # first, convert image to PyTorch tensor
                tt.Normalize((0.1307,), (0.3081,)) # normalize inputs
            ])
        ),
        batch_size=512,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    val_dl = DataLoader(
        datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=tt.Compose([
                tt.ToTensor(), # first, convert image to PyTorch tensor
                tt.Normalize((0.1307,), (0.3081,)) # normalize inputs
            ])
        ),
        batch_size=32,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )

    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    return train_dl, val_dl

class RegressionCoinDataset(Dataset):
    def __init__(self, directory, transform=False):
        self.directory = directory
        self.transform = transform
        self.files = os.listdir(self.directory)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        label = float(name.split('_')[0]) / 100.0

        im = read_image(os.path.join(self.directory, name))
        if self.transform:
            im = self.transform(im)

        return im, label

def load_br_coins(
    path='/container_app/br-coins', 
    transforms=tt.Compose([
            tt.Resize((240, 320)), antialias=True
        ]),
    device=get_default_device):

    train_ds = RegressionCoinDataset(
        os.path.join(path, 'train'), 
        transforms
    )
    test_ds = RegressionCoinDataset(
        os.path.join(path, 'test'), 
        transforms
    )

    train_dl = DeviceDataLoader(
        DataLoader(
            train_ds, 
            batch_size=12, 
            shuffle=True, 
            num_workers=16, 
            pin_memory=True
        ), 
        device
    )
    test_dl = DeviceDataLoader(
        DataLoader(
            test_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=16, 
            pin_memory=True
        ), 
        device
    )

    return train_dl, test_dl