# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:49:44 2024

@author: User
"""

import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def binary_accuracy(predictions, labels):
    rounded_predictions = torch.round(predictions)
    correct = (rounded_predictions == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy



def create_sliding_windows(data, window_size, stride=1):
    windows = []
    data_length = len(data)

    for i in range(0, data_length - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)

    return np.array(windows)

data1 = np.load('AI-data 1.npy')
data2 = np.load('Human-data 1.npy')


window_size = 100
stride = 1

AIdata = create_sliding_windows(data1, window_size, stride)
Humandata = create_sliding_windows(data2, window_size, stride)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1, :])
        out = self.sigmoid(out)
        return out

AIdata=torch.from_numpy(AIdata)
Humandata=torch.from_numpy(Humandata)
merged_data = torch.cat((AIdata, Humandata), dim=0)


labels_data1 = torch.ones((AIdata.size(0), 1)) 
labels_data2 = torch.zeros((Humandata.size(0), 1))  
y = torch.cat((labels_data1, labels_data2), dim=0).float()
#print(AIdata.size())
#print(Humandata.size())
#print(y)

input_size = 2  # 每個時間步的特徵數
hidden_size = 64  # LSTM隱藏層的大小
output_size = 1  # 二分類，輸出的維度為1
num_layers = 1  # LSTM的層數

model = LSTMClassifier(input_size, hidden_size, output_size, num_layers)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


merged_data = torch.tensor(merged_data, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

epochs = 30

for epoch in range(epochs):
    model.train()  # 將模型設置為訓練模式
    optimizer.zero_grad()
    outputs = model(merged_data)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()


    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


    model.eval()  
    with torch.no_grad():
        val_outputs = model(merged_data)
        val_accuracy = binary_accuracy(val_outputs, y)


    print(f'Validation Accuracy: {val_accuracy.item()*100:.2f}%')
    
model.eval()  
with torch.no_grad():
    final_outputs = model(merged_data)
    final_predictions = torch.round(final_outputs)
    y_true = y.numpy()
    y_pred = final_predictions.numpy()


cm = confusion_matrix(y_true, y_pred)


plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()