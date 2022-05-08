import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import torch
from torch import random
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch.nn as nn
from torchvision.transforms import Resize, Normalize
from torch.optim import Adam, SGD

df = pd.read_csv('./preprocessed_blocking_data.csv')

event_ids = df['event_id'].to_numpy()
y_data = df['block_intensity'].to_numpy() 

x_reanalysis_data = []

print("Loading reanalysis data...")
for event_id in event_ids:
    event_array = np.load(f"./reanalysis_data_processed/{event_id}.npy")
    x_reanalysis_data.append(event_array)

x_reanalysis_data = np.array(x_reanalysis_data)


tensor_event_ids = torch.Tensor(event_ids)
tensor_x = torch.Tensor(x_reanalysis_data)
tensor_y = torch.Tensor(y_data)
tensor_y = tensor_y.view(-1,1)

resizer = Resize((64,128)) #bilinear interpolation
tensor_x_transformed = resizer(tensor_x)
# print(tensor_x_transformed.shape)


dataset = TensorDataset(tensor_event_ids, tensor_x_transformed, tensor_y)
test_size= int(round(len(dataset)*0.3))
train_size= len(dataset) - test_size

train_set, test_set= random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0)) #Splits same way every time

# normalizer= Normalize(mean = torch.mean(tensor_x))



BATCH_SIZE= 64
train_data_loader = DataLoader(train_set, batch_size = BATCH_SIZE)
test_data_loader = DataLoader(test_set, batch_size= BATCH_SIZE)

# print(len(train_set))
# print(len(train_data_loader))
# for batch in train_data_loader:
#     ids, x, y = batch
#     print(ids.shape) # 64 samples
#     print(x.shape) # 64 x image shape
#     print(y.shape) # 64 samples

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(17, 32, 3, padding="same"),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm2d(32),
                                  nn.MaxPool2d(2,2), #kernel_size=2, stride=2, 32,64,128 -> 32,32,64
                                  nn.Dropout2d(p=0.4),
                                  nn.Conv2d(32, 32, 3, padding="same"),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm2d(32),
                                  nn.MaxPool2d(2,2), # 32,32,64 -> 32,16,32
                                  nn.Dropout2d(p=0.4),
                                  nn.Conv2d(32, 32, 3, padding="same"),
                                  nn.LeakyReLU(),
                                  nn.BatchNorm2d(32),
                                  nn.MaxPool2d(2,2),
                                  nn.Dropout2d(p=0.1)
                                  ) #32,16,32 -> 32,8,16

        self.mlp = nn.Sequential(nn.Linear(32*8*16,512), 
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(512),
                                 nn.Dropout(p=0.1),
                                 nn.Linear(512,256),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(256),
                                 nn.Dropout(p=0.1),
                                 nn.Linear(256,64),
                                 nn.LeakyReLU(),
                                 nn.BatchNorm1d(64),
                                 nn.Linear(64,1))
    
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out_flat = torch.nn.Flatten()(conv_out)
        pred = self.mlp(conv_out_flat)
        return pred

model = ConvModel()
print(model) 

def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target))

loss_function = nn.L1Loss() #aka Mean Absolute Error
loss_function_mape = MAPELoss
optimizer= Adam(model.parameters(), lr = 1e-3)

def train_step(data_loader):
    model.train()
    train_loss = 0
    train_mape =  0
    for batch_idx, batch in enumerate(data_loader):
        ids, x, y = batch
        pred = model(x)

        loss = loss_function(pred, y)
        mape = loss_function_mape(pred,y)
        train_loss = loss.item() + train_loss
        train_mape= mape.item() + train_mape
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = train_loss/len(data_loader)
    train_mape = train_mape/len(data_loader)
    print(f"train_loss: {train_loss}")
    print(f"train_mape: {train_mape}")


t_test_dataset= pd.DataFrame()

def test_step(data_loader):
    model.eval()
    test_loss = 0
    test_mape = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            ids, x, y = batch
            pred = model(x)
            loss = loss_function(pred, y)
            mape = loss_function_mape(pred,y)
            test_loss = loss.item() + test_loss
            test_mape= mape.item() + test_mape
            t_test_dataset['prediction'] = pd.Series(pred.flatten())
            t_test_dataset['actual'] = pd.Series(y.flatten())
            print(t_test_dataset)
            path = '/home/emirs/blocking-research/'
            t_test_dataset.to_csv(os.path.join(path,fr't_test_dataset_cnn_{batch_idx}.csv'),index=False)
    test_loss= test_loss/len(data_loader)
    test_mape= test_mape/len(data_loader)
    print(f"test_loss: {test_loss}")
    print(f"test_mape: {test_mape}")
        

EPOCHS = 50
for i in range(EPOCHS):
    print(f"Epoch {i}")
    print(f"------------------")
    train_step(train_data_loader)
    test_step(test_data_loader)
    print()






    