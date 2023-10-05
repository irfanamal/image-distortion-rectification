import cv2
import os
import pandas as pd
import torch
import torch.nn as nn

from time import time
from transform import ImageRectification
from tqdm import tqdm
from utils import val_loader

test_dataset, test_data_loader = val_loader('images/test/distorted', 'images/test/test.csv', shuffle=False, batch_size=1, num_workers=2)

experiment = 'exp_1'
os.makedirs(f'images/test/rectified/{experiment}', exist_ok=True)

model = torch.load(f'models/{experiment}.pth').eval().cuda()
metric = nn.L1Loss().cuda()

start = time()
model.eval()
test_loss = 0
results = []
transform = ImageRectification()
with tqdm(test_data_loader, leave=False) as test_loop:
    with torch.no_grad():
        for i, (feature, k) in enumerate(test_loop):
            feature = feature.cuda()
            k = k.cuda()
            output = model(feature)

            mae = metric(output, k)
            test_loss += mae.item()

            image_file = test_dataset.annotation.loc[i, 'image']
            results.append([image_file, output.item()])

            image = cv2.imread(f'images/test/distorted/{image_file}')
            image = transform.rectify(image, output.item())
            cv2.imwrite(f'images/test/rectified/{experiment}/{image_file}', image)

test_loss /= len(test_dataset)
test_time = (time() - start)/len(test_dataset)

df = pd.DataFrame(columns=['image', 'k'], data=results)
df.to_csv(f'results/{experiment}.csv', index=False)

with open(f'results/{experiment}.txt', 'a+') as f:
    f.write(f'test_loss: {test_loss} (Mean Absolute Error)\n')
    f.write(f'test_time: {test_time} (Seconds/Input AVG)\n')
print(f'MAE: {test_loss}')
print(f'Test Time: {test_time * len(test_dataset)}')