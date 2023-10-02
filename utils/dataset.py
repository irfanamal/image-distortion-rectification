import cv2
import numpy as np
import os
import pandas as pd
import random
import torch

from torch.utils.data import Dataset
from transform.gaussian import InvertedGaussian
from typing import Callable

class DistortedImageDataset(Dataset):
    def __init__(self, image_dir: str, annotation: str, transform: Callable = None, distort: Callable = None) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.distort = distort
        self.transform = transform
        self.annotation = pd.read_csv(annotation)

    def __len__(self) -> int:
        return self.annotation.shape[0]
    
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(os.path.join(self.image_dir, self.annotation.loc[index, 'image']))

        image_tensor = torch.from_numpy(np.moveaxis(image, -1, 0))

        if self.transform:
            image_tensor = self.transform(image_tensor)

        image = np.moveaxis(image_tensor.numpy(), 0, -1)

        k = None
        if self.distort:
            k = round(random.random(), 2)
            image = self.distort(image, k)
        else:
            k = self.annotation.loc[index, 'k']

        edge_map = np.expand_dims(cv2.Canny(image, 255/3, 255).astype(np.float32), axis=-1)/255

        weight_map = InvertedGaussian().generate(image)
        image = (image.astype(np.float32) / 255) + weight_map

        feature = np.concatenate((image, edge_map), axis=-1)
        feature = np.moveaxis(feature, -1, 0)
        feature = torch.from_numpy(feature)

        return feature, torch.Tensor([k])