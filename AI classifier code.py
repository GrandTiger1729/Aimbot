# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:49:44 2024

@author: User
"""

import torch
import numpy as np
import torch.nn as nn

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(200,100)
        self.layer2=nn.Linear(100,2)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        x=self.softmax(x)
        return x

aiData=np.load('data/AI data/AI-data 1.npy')
humanData=np.load('data/Human data/Human-data 1.npy')

counterAI=NN()

optimizer=torch.optim.Adam(counterAI.parameters())
lr=0.01
loss_function=nn.CrossEntropyLoss()


print(aiData[:20])
print(humanData[:20])