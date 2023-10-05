import os
import torch
import torch.nn as nn

from correction import RDCCorrectionNet
from time import time
from tqdm import tqdm
from utils import train_loader, val_loader

train_dataset, train_data_loader = train_loader('images/train/original', 'images/train/train.csv', batch_size=1, num_workers=2)
val_dataset, val_data_loader = val_loader('images/val/distorted', 'images/val/val.csv', batch_size=1, num_workers=2)

epochs = 20
experiment = 'exp_2'
os.makedirs(f'ckpt/{experiment}', exist_ok=True)

model = RDCCorrectionNet().cuda()
criterion = nn.MSELoss().cuda()
metric = nn.L1Loss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

best_train_loss = float('inf')
best_val_loss = float('inf')
patience = 2
convergence = False
for epoch in range(epochs):
    epoch_train_loss = 0
    epoch_val_loss = 0

    start = time()
    model.train()
    with tqdm(train_data_loader, f'Epoch {epoch} Train', leave=False) as train_loop:
        for feature, k in train_loop:
            feature = feature.cuda()
            k = k.cuda()
            output = model(feature)

            loss = criterion(output, k)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += (loss.item() * feature.size(0))

            train_loop.set_postfix({'Train Loss': loss.item()})
    epoch_train_loss /= len(train_dataset)
    train_time = time() - start
    
    start = time()
    model.eval()
    with tqdm(val_data_loader, f'Epoch {epoch} Val', leave=False) as val_loop:
        with torch.no_grad():
            for feature, k in val_loop:
                feature = feature.cuda()
                k = k.cuda()
                output = model(feature)

                loss = metric(output, k)
                epoch_val_loss += (loss.item() * feature.size(0))

                val_loop.set_postfix({'Val Loss': loss.item()})
    epoch_val_loss /= len(val_dataset)
    val_time = time() - start

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': best_train_loss,
                'val_loss': best_val_loss},
                f'ckpt/{experiment}/{epoch}.pt')

    if epoch_val_loss < best_val_loss:
        best_train_loss = epoch_train_loss
        best_val_loss = epoch_val_loss
        if not convergence:
            patience = 2
        
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': best_train_loss,
                    'val_loss': best_val_loss},
                    f'ckpt/{experiment}/best.pt')
    else:
        patience -= 1
    
    if patience == 0:
        convergence = True
        optimizer.param_groups[0]['lr'] /= 10

    with open(f'logs/{experiment}.log', 'a+') as f:
        f.write(f'Epoch: {epoch}, train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}, train_time: {train_time}, val_time: {val_time}\n')
    print(f'Epoch: {epoch}, train_loss: {epoch_train_loss}, val_loss: {epoch_val_loss}, train_time: {train_time}, val_time: {val_time}')

checkpoint = torch.load(f'ckpt/{experiment}/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
torch.save(model, f'models/{experiment}.pth')