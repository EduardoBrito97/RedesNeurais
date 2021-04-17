import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 256

class Dataset(data.Dataset):
    
    def __init__(self, path_atributes, path_labels):

        self.labels = pd.read_csv(path_labels, index_col=0).values
        self.atributes = StandardScaler().fit_transform(pd.read_csv(path_atributes, index_col=0).values)

    def __getitem__(self, index):
        
        return torch.Tensor(self.atributes[index].astype(float)), self.labels[index]

    def __len__(self):

        return self.atributes.shape[0]

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.fc1 = nn.Linear(305, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, x):

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x
    
train_dataset_loader = Dataset('../data/X_train_over.csv', '../data/y_train_over.csv')
train_dataset = data.DataLoader(dataset=train_dataset_loader, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

valid_dataset_loader = Dataset('../data/X_valid.csv', '../data/y_valid.csv')
valid_dataset = data.DataLoader(dataset=valid_dataset_loader, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

net = Net().to(DEVICE)
optmizer = optim.Adam(net.parameters(), lr=0.0005)
critirion = nn.MSELoss()

print('TRAIN:')
for epoch in range(200):

    net.train()
    
    train_loss = 0
    train_accuracy = 0

    valid_loss = 0
    valid_accuracy = 0

    for X, y in train_dataset:

        optmizer.zero_grad()

        X, y = X.to(DEVICE), y.to(DEVICE)
        output = net(X)

        loss = critirion(output.float(), y.float())
        
        train_loss += loss.item()
        train_accuracy += accuracy_score(output.view(BATCH_SIZE).detach().numpy().round(), y.detach().numpy().round())

        loss.backward()
        optmizer.step()

    net.eval()
    with torch.no_grad():
        for X, y in valid_dataset:
            
            X, y = X.to(DEVICE), y.to(DEVICE)
            output_valid = net(X)
            
            valid_loss += critirion(output_valid.float(), y.float()).item()
            valid_accuracy += accuracy_score(output_valid.view(BATCH_SIZE).detach().numpy().round(), y.detach().numpy().round())

    print(f'EPOCH: {(epoch + 1):03} | TRAIN_LOSS: {train_loss/len(train_dataset):.5f} | TRAIN_ACCURACY: {train_accuracy/len(train_dataset):.5f} | VALID_LOSS: {valid_loss/len(valid_dataset):.5f} | VALID_ACCURACY: {valid_accuracy/len(valid_dataset):.5f}')
