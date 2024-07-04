
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


    train_dl, val_dl = load_br_coins(device=device)

    model = to_device(get_vision_transformer(ImageRegressionBase)(img_size=(320,240), patch_size(4,4), n_channel=3, blocks=2), device)

    history = fit(
        epochs=100,
        lr=0.05,
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        num=0
    )

    plot_regression(model, val_dl, name='./0_regression')
    plot_losses(history, name='./0_regression_losses')