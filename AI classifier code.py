# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:49:44 2024

@author: User
"""
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
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

data1 = np.load('AI-data 1.npy')
data2 = np.load('Human-data 1.npy')

window_size = 100
stride = 1

AIdata = create_sliding_windows(data1, window_size, stride)
Humandata = create_sliding_windows(data2, window_size, stride)

AIdata = torch.from_numpy(AIdata)
Humandata = torch.from_numpy(Humandata)


labels_data1 = torch.ones((AIdata.size(0), 1))
labels_data2 = torch.zeros((Humandata.size(0), 1))
y = torch.cat((labels_data1, labels_data2), dim=0).float()

merged_data = torch.cat((AIdata, Humandata), dim=0)
indices = np.arange(len(merged_data))
np.random.shuffle(indices)
merged_data = merged_data[indices]
y = y[indices]


train_size = int(0.8 * len(merged_data))
train_data, test_data = merged_data[:train_size], merged_data[train_size:]
train_labels, test_labels = y[:train_size], y[train_size:]


input_size = 2
hidden_size = 64
output_size = 1
num_layers = 1

model = LSTMClassifier(input_size, hidden_size, output_size, num_layers)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

epochs = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        val_outputs = model(test_data)
        val_accuracy = binary_accuracy(val_outputs, test_labels)

    print(f'Testing Accuracy: {val_accuracy.item()*100:.2f}%')

# Confusion Matrix for Testing Data
model.eval()
with torch.no_grad():
    final_outputs = model(test_data)
    final_predictions = torch.round(final_outputs)
    y_true = test_labels.numpy()
    y_pred = final_predictions.numpy()

cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(f'Testing Confusion Matrix:\n{cm}')
print(f'Testing Accuracy: {accuracy*100:.2f}%')

# Visualize Confusion Matrix
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Human', 'Predicted Cheat'],
            yticklabels=['Actual Human', 'Actual Cheat'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Testing Confusion Matrix')
plt.show()