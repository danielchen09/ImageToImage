import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir, test=False):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.input_transforms = [A.Resize(width=256, height=256)]
        if not test:
            self.input_transforms += [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.2)
            ]
        self.input_transforms += [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2()
        ]
        self.input_transforms = A.Compose(self.input_transforms)
        self.target_transforms = A.Compose([
            A.Resize(width=256, height=256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = np.array(Image.open(os.path.join(self.root_dir, self.files[index])))
        in_img = self.input_transforms(image=img[:, :600, :])['image']
        target_img = self.target_transforms(image=img[:, 600:, :])['image']
        return in_img, target_img


if __name__ == "__main__":
    dataset = MapDataset("../data/maps/train/", test=True)
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "imgs1/x.png")
        save_image(y, "imgs1/y.png")
        break