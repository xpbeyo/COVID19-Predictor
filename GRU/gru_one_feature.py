import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Training and validation Dataset for DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, file_name, window_size, train=True, test=False):
        data = pd.read_csv(file_name).values[1:, 1:]    # remove title row and date column
        data = np.array(data, dtype=np.float32)
        data[np.isnan(data)] = 0    # Mask NaN with zero
        # Use data from previous day(s) to predict the confirmed cases of next day
        mark = int(np.round(data.shape[0]*0.8))     # 80% mark of dataset

        X = []
        y = []

        if not test:
            for i in range(mark - window_size - 1):     
                X.append(data[i:window_size+i, 0].reshape(-1, 1))   # Each example is (window_size * n_features)
                y.append(data[window_size+i, 0])        # Each target is scaler (cases of the next date)

            split = int(np.round(len(y)*0.7))           # Split train and val 7:3
            
            if train:
                self.X = torch.tensor(X[:split], dtype=torch.float32)
                self.y = torch.tensor(y[:split], dtype=torch.float32)
            else:
                self.X = torch.tensor(X[split:], dtype=torch.float32)
                self.y = torch.tensor(y[split:], dtype=torch.float32)
        
        else: 
            for i in range(mark, data.shape[0]-window_size-1):
                X.append(data[i:window_size+i, 0].reshape(-1, 1))
                y.append(data[window_size+i, 0])
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.y[index] 
    
    def __len__(self):
        return len(self.y)


# GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Input to network have shape (batch_size, seq_len, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Randomly initialized hidden states
        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device) 
        
        out, _ = self.gru(x, h0)    # shape = (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]         # Output of the last GRU cell
        out = self.dense(out)       # shape = (batch_size, 1, 1)
        out = self.relu(out)

        return out


def train(model, train_loader, val_loader, num_epochs, learning_rate, device):

    criterion = nn.MSELoss()    # Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer

    # Main training loop
    training_losses = []
    val_losses = []
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        for i, (sample, target) in enumerate(train_loader):  
            sample = sample.to(device)
            target = target.to(device)      # shape = (batch_size, 1)

            # Forward pass
            outputs = model(sample)         # shape = (batch_size, 1, 1)

            if epoch == num_epochs - 1:
                # print(sample)
                print('Training prediction')
                print(outputs.squeeze())
                print('Training target')
                print(target)

            loss = criterion(outputs.squeeze(), target)     # MSE loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())

            val_loss = validation_loss(model, val_loader, device, epoch)
            val_losses.append(val_loss)

            if epoch % 100 == 0:
                # print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], '
                #         f'Training Loss: {loss.item():.4f}')
                print (f'Epoch {epoch+1:5}/{num_epochs:5}     Batch {i+1}/{num_batches}     '
                        f'Training Loss: {loss.item():9.3f}    Valid Loss: {val_loss:8.3f}')

    plt.figure()
    plt.plot(np.arange(len(training_losses))[2500:], training_losses[2500:], color='b')
    plt.plot(np.arange(len(training_losses))[2500:], val_losses[2500:], color='r')

    plt.show()


def validation_loss(model, val_loader, device, epoch):
    # Calculate validation loss
    criterion = nn.MSELoss()

    with torch.no_grad():
        for sample, target in val_loader:
            sample = sample.to(device)
            target = target.to(device)          # shape (val_set_size, 1)
            prediction = model(sample)          # shape (val_set_size, 1, 1)

            # Round function has zero gradient, so apply it when calculating val loss
            rounded_pred = torch.round(prediction.squeeze())    
            loss = criterion(rounded_pred, target)

            if epoch % 2000 == 0:
                print('Rounded validation sample prediction')
                print(rounded_pred)
                print('Validation target')
                print(target)

    if epoch == -1: # Evaluation only in test set
        return loss.item(), rounded_pred, target

    return loss.item()


if __name__ == '__main__':
    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters 
    num_epochs = 12000      # 12000 optimal (10000?)
    batch_size = 245        # 245 (full batch)
    val_batch_size = 200    # full batch (actual size < 200)
    learning_rate = 0.0003   # 0.001
    input_size = 1          # 1     # num_features (fixed)
    seq_len = 11             # 7        # window_size
    hidden_size = 512       # 512
    num_layers = 2


    # Load data for training
    training_set = TimeSeriesDataset('covid19_sg_clean_reduced.csv', seq_len, train=True)
    val_set = TimeSeriesDataset('covid19_sg_clean_reduced.csv', seq_len, train=False)
    train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=val_batch_size, shuffle=True)

    model = GRU(input_size, hidden_size, num_layers).to(device)

    train(model, train_loader, val_loader, num_epochs, learning_rate, device)

    # Load test set and evaluate
    test_set = TimeSeriesDataset('covid19_sg_clean_reduced.csv', seq_len, test=True)
    test_loader = DataLoader(dataset=test_set, batch_size=200, shuffle=False)
    test_mse, pred, target = validation_loss(model, test_loader, device, -1)
    print(f'\n\nTest set rmse: {np.sqrt(test_mse)}\n')


    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    plt.figure()
    plt.plot(np.arange(len(pred)), pred, 'r', label='Prediction')
    plt.plot(np.arange(len(pred)), target, 'b', label='Target')
    plt.legend()
    plt.show()