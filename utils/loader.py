from utils import DistortedImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transform.distortion import ImageDistortion
from typing import Callable

def get_loader(image_dir: str, annotation: str, transform: Callable, distort: Callable, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset = DistortedImageDataset(image_dir=image_dir, annotation=annotation, transform=transform, distort=distort)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset, data_loader

def train_loader(image_dir: str, annotation: str, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    transformation = transforms.Compose([transforms.Resize(256, antialias=True), transforms.RandomCrop(256), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    distortion = ImageDistortion().distort
    return get_loader(image_dir=image_dir, annotation=annotation, transform=transformation, distort=distortion, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def val_loader(image_dir: str, annotation: str, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    transformation = transforms.Compose([transforms.Resize(256, antialias=True)])
    return get_loader(image_dir=image_dir, annotation=annotation, transform=transformation, distort=None, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)