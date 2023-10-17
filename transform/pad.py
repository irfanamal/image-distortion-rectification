import torch

from torchvision import transforms

class Pad():
    def __init__(self, size: tuple) -> None:
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        pad_h = self.size[0] - h
        pad_w = self.size[1] - w

        pad_h_1 = (pad_h) // 2
        pad_h_2 = pad_h - pad_h_1
        pad_w_1 = (pad_w) // 2
        pad_w_2 = pad_w - pad_w_1

        pad = transforms.Pad((pad_w_1, pad_h_1, pad_w_2, pad_h_2))
        image = pad(img)
        return image