import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt


input_size = 3  # Number of input features
hidden_size = 50  # Number of LSTM units
output_size = 1  # Number of output features
num_layers = 2  # Number of LSTM layers
num_epochs = 100
learning_rate = 0.001
sequence_length = 10
batch_size = 32

directional = True # will only train on pucks moving in the correct direction

def create_sequences(x, y, seq_length):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    x = torch.mul(x,1)
    y = torch.mul(y,1)
    sequences = []
    labels = []

    for i in range(len(x) - seq_length):
        if x[i+seq_length][2] != -1: # if direction != left
            continue
        if x[i+seq_length][0] < 0: # if past goal
            continue

        seq = x[i:i+seq_length]
        label = y[i+seq_length]
        sequences.append(seq)
        labels.append(label)

    return torch.stack(sequences), torch.stack(labels)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out
    
## main
df = pd.concat([pd.read_csv("Data/cleaned_data.csv"), pd.read_csv("Data/cleaned_data_flipped.csv")], ignore_index=True)

X = df[["Puck_cen_X","Puck_cen_Y","direction"]].values # values between 2 and -1
Y = df[["Cross_Left"]].values
#Y = df["Cross_right"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

train_sequences, train_labels = create_sequences(X_train, y_train, sequence_length)
val_sequences, val_labels = create_sequences(X_val, y_val, sequence_length)
test_sequences, test_labels = create_sequences(X_test, y_test, sequence_length)
del X, Y, X_train, X_test, X_val, y_train, y_test, y_val # deleate

#create Model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
optimizer = optim.Adam(model.parameters())
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#dataloaders
train_loader = DataLoader((train_sequences, train_labels), batch_size=batch_size, shuffle=False)
val_loader = DataLoader((val_sequences, val_labels), batch_size=batch_size, shuffle=False)
train_loader = DataLoader((test_sequences, test_labels), batch_size=batch_size, shuffle=False)

#arrays
train_loss =np.zeros(num_epochs)
val_loss=np.zeros(num_epochs)

#training
for epoch in range(num_epochs):
    outputs = model(train_sequences)
    loss = criterion(outputs, train_labels)
    train_loss[epoch] = loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs_val = model(val_sequences)
    loss_val = criterion(outputs_val, val_labels)
    val_loss[epoch] = loss_val

    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



plt.plot(train_loss, label = 'train loss')
plt.plot(val_loss, label = "val loss")
plt.legend
plt.show


#testing
model.eval()
mseLoss = nn.MSELoss()
maeLoss = nn.L1Loss()
with torch.no_grad():
    test_outputs = model(test_sequences)
    mse = mseLoss(test_outputs, test_labels)
    mae = maeLoss(test_outputs, test_labels)
    print(f'Mean Square Error on Test Data: {mse.item():.4f}')
    print(f'Mean Absolute Error on Test Data: {mae.item():.4f}')

plt.plot(test_labels.numpy(), label='Actual')
plt.plot(test_outputs.numpy(), label='Predicted')
plt.legend()
plt.show()