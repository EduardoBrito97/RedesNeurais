#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# In[ ]:


def labelizer(labels):
    result = torch.empty(len(labels), 10)
    for idx, label in enumerate(labels):
        aux = torch.zeros(1, 10)
        aux[0][label] = 1
        result[idx]= aux
    
    return result


# In[ ]:


def get_train_output(outputs, labels):
    result = torch.empty(len(outputs), 1)
    
    for idx, (out, label) in enumerate(zip(outputs, labels)):
        result[idx] = out[label]
    
    return result

def get_test_output(outputs, labels):
    result = torch.empty(len(outputs), 1)
    
    for idx, (out, label) in enumerate(zip(outputs, labels)):
        out_list = out.tolist()
        result[idx] = 1 if out_list.index(max(out_list)) == label.item() else 0
     
    return result


# In[ ]:


class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=2, padding=0, stride=2),
            torch.nn.ReLU(inplace=True),
        )
        
        self.linear_block1 = torch.nn.Sequential(
            torch.nn.Linear(64*14*14, 1024),
            torch.nn.ReLU(inplace=True),
        )
        
        self.linear_block2 = torch.nn.Sequential(
            torch.nn.Linear(1024, 10),
            torch.nn.Softmax(dim=1),
        )
        
    def forward(self, input):
        x = self.conv_block(input)
        x = x.view(x.size(0), -1)
        x = self.linear_block1(x)
        x = self.linear_block2(x)
        
        return x


# In[ ]:


def train_loop(train_data, batch_size):
    net.train()
    running_loss = 0
    
    for idx, (input, label) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        out = net(input.to(DEVICE))
        prob_out = get_train_output(out, label)
        loss = loss_fn(prob_out.to(DEVICE), torch.ones(len(out), 1).to(DEVICE))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx > 32000/batch_size:
            break
    return running_loss / len(train_data)


# In[ ]:


def test_loop(test_data):
    net.eval()
    running_loss = 0
    hits = 0
    out_size = 0
    
    for idx, (input, label) in enumerate(train_loader, start=1):
        out = net(input.to(DEVICE))
        prob_out = get_train_output(out, label)
        binary_out = get_test_output(out, label)
        loss = loss_fn(prob_out.to(DEVICE), torch.ones(len(out), 1).to(DEVICE))
        
        running_loss += loss.item()
        hits += torch.sum(binary_out).item()
        out_size += len(binary_out)
        if idx > 4:
            break
    return hits / out_size


# In[ ]:


def get_data(batch_size):
  train_loader = torch.utils.data.DataLoader(torchvision.datasets.QMNIST('./',
                                                                        download=True, 
                                                                        train=True, 
                                                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=2,
                                            )

  test_loader = torch.utils.data.DataLoader(torchvision.datasets.QMNIST('./',
                                                                        download=True, 
                                                                        train=False, 
                                                                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])),
                                            batch_size=1024,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=2,
                                            )
  
  return train_loader, test_loader


# In[ ]:


BATCHES = [16, 32, 64, 128]
LEARNING_RATES = [0.01, 0.1, 1]
MOMENTUMS = [0.01, 0.1, 1]
writer = SummaryWriter()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
out_dicts = []
for bs in BATCHES:
  train_loader, test_loader = get_data(bs)
  #print(train_loader.shape)
  #print(test_loader.shape)
  for lr in LEARNING_RATES:
    for mom in MOMENTUMS:
      net = model().to(DEVICE)
      loss_fn = torch.nn.MSELoss()
      optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=mom)
      
      training_loss = []
      training_acc = []
      start = time.time()
      tmp_dict = {}
      for epoch in range(30):
        loss = train_loop(train_loader, bs)
        training_loss.append(loss)
        writer.add_scalars('loss_lr'+str(lr)+'_mom'+str(mom), {'batch_size '+str(bs):loss}, epoch+1)

        test_acc = test_loop(test_loader)
        writer.add_scalars('test_acc_lr'+str(lr)+'_mom'+str(mom), {'batch_size '+str(bs):test_acc}, epoch+1)
        training_acc.append(test_acc)
        
        run_time = time.time() - start

        print(f'LR: {lr}, MOM: {mom}, BS: {bs}')
        print(f'{epoch} epochs  in {run_time} sec')
        print(f'Accuracy: {max(training_acc)}')
        print('----------------------------------------------')
        
        tmp_dict['LR'] = lr
        tmp_dict['MOM'] = mom
        tmp_dict['BS'] = bs
        tmp_dict['Epochs'] = epoch
        tmp_dict['Acc'] = max(training_acc)
        tmp_dict['Time'] = run_time
        out_dicts.append(tmp_dict)
        if max(training_acc) > (1- 1e-3):
            break
writer.close()


# In[ ]:


df = pd.DataFrame.from_dict(out_dicts)
df.head()
df.to_csv('out.csv')

