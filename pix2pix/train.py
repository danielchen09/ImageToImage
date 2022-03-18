import torch
from Discriminator import Discriminator
from Generator import Generator
from dataset import MapDataset
import config
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(disc, gen, train_loader, opt_d, opt_g, l1_loss, bce_loss, d_scalar, g_scalar):
    for step, (x, y) in enumerate(tqdm(train_loader)):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            d_real = disc(x, y)
            d_real_loss = bce_loss(d_real, torch.ones_like(d_real))
            d_fake = disc(x, y_fake.detach())
            d_fake_loss = bce_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2

        opt_d.zero_grad()
        d_scalar.scale(d_loss).backward()
        d_scalar.step(opt_d)
        d_scalar.update()

        with torch.cuda.amp.autocast():
            d_fake = disc(x, y_fake)
            g_fake_loss = bce_loss(d_fake, torch.ones_like(d_fake))
            g_l1_loss = l1_loss(y_fake, y) * config.L1_LAMBDA
            g_loss = g_fake_loss + g_l1_loss

        opt_g.zero_grad()
        g_scalar.scale(g_loss).backward()
        g_scalar.step(opt_g)
        g_scalar.update()


def main():
    disc = Discriminator().to(config.DEVICE)
    gen = Generator().to(config.DEVICE)
    opt_d = torch.optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    if config.LOAD_MODEL:
        load_checkpoint(config.GEN_CKPT, gen, opt_g, config.LEARNING_RATE)
        load_checkpoint(config.DISC_CKPT, disc, opt_d, config.LEARNING_RATE)

    train_ds = MapDataset(config.TRAIN_DIR)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    l1_loss = torch.nn.L1Loss()

    g_scalar = torch.cuda.amp.GradScaler()
    d_scalar = torch.cuda.amp.GradScaler()

    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train(disc, gen, train_loader, opt_d, opt_g, l1_loss, bce_loss, d_scalar, g_scalar)
        if epoch % 10 == 0:
            save_checkpoint(config.GEN_CKPT, gen, opt_g)
            save_checkpoint(config.DISC_CKPT, disc, opt_d)

        save_some_examples(gen, val_loader, epoch, folder='imgs')


if __name__ == '__main__':
    main()