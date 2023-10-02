import cv2
import os
import pandas as pd
import random

from joblib import Parallel, delayed
from transform.distortion import ImageDistortion

def create_csv(split: str, data: list) -> None:
    df = None
    if split == 'train':
        df = pd.DataFrame(columns=['image'], data=data)
    else:
        df = pd.DataFrame(columns=['image', 'k'], data=data)
    df.to_csv(f'images/{split}/{split}.csv', index=False)

def batch_distort(split: str) -> list:
    distorter = ImageDistortion()
    def distort(file: str) -> tuple[str, float]:
        image = cv2.imread(f'images/{split}/original/{file}')
        k = round(random.random(), 2)
        image = distorter.distort(image, k)
        cv2.imwrite(f'images/{split}/distorted/{file}', image)
        return file, k
    
    data = Parallel(n_jobs=-1)(delayed(distort)(file) for file in os.listdir(f'images/{split}/original'))
    return data

def main():
    splits = ['train', 'val', 'test']
    for split in splits:
        data = None
        if split == 'train':
            data = os.listdir(f'images/{split}/original')
        else:
            data = batch_distort(split)
        create_csv(split, data)

main()