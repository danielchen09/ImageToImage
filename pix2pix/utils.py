import torch
from torchvision.utils import save_image
from pix2pix import config


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        if epoch <= 20:
            save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
    gen.train()


def save_checkpoint(path, model, optimizer):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, lr):
    ckpt = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr