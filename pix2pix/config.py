import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
IMG_CHANNELS = 3
L1_LAMBDA = 100
NUM_EPOCHS = 1000
LOAD_MODEL = False
SAVE_MODEL = True
GEN_CKPT = 'checkpoints/gen.pth.tar'
DISC_CKPT = 'checkpoints/disc.pth.tar'
TRAIN_DIR = '../data/maps/train'
VAL_DIR = '../data/maps/val'