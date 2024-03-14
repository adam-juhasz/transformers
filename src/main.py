
from utils import *
from network import ImageClassificationBase, ImageRegressionBase, get_vision_transformer

if __name__ == '__main__':
    device = get_default_device()
    
    train_dl, val_dl = load_mnist(device=device)

    model = to_device(get_vision_transformer(ImageClassificationBase)(), device)

    history = fit(
        epochs=100, 
        lr=0.001, 
        model=model, 
        train_loader=train_dl, 
        val_loader=val_dl
    )

    plot_losses(history)
    plot_accuracies(history)
