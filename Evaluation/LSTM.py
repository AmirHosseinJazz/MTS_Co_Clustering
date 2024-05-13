import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_univariate(data, target_feature_index, n_lags=5):
    """
    Preprocess the data for univariate time series prediction.
    
    Parameters:
    data (numpy array): The original dataset of shape (samples, timesteps, features).
    target_feature_index (int): The index of the target feature.
    n_lags (int): The number of lagged observations to use for prediction.
    
    Returns:
    X (numpy array): Input features of shape (samples, n_lags, 1).
    y (numpy array): Target values of shape (samples, 1).
    """
    n_samples, n_timesteps, n_features = data.shape
    X = []
    y = []
    
    for sample in range(n_samples):
        for t in range(n_lags, n_timesteps):
            X.append(data[sample, t-n_lags:t, target_feature_index].reshape(-1, 1))
            y.append(data[sample, t, target_feature_index])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,default="TimeGAN_model1", help='Name of the model')
    parser.add_argument('--window', type=int, default=10, help='Window size')
    parser.add_argument('--target_feature_index', type=int, default=2, help='Target feature index')
    args = parser.parse_args()


    model_name = args.model
    window = args.window
    target_feature_index = args.target_feature_index

    np.random.seed(42)
    # Simulated real data (180 cases, 120 time steps, 30 features)

    real_data = np.random.rand(100, 120, 30)
    # Select a target feature index (e.g., 2)

    generated_data= np.random.rand(100, 120, 30)

    # Preprocess the data
    X, y = preprocess_univariate(real_data, target_feature_index, n_lags=window)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Convert to TensorDataset
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).unsqueeze(1))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    ###### Print the shapes of the datasets ######
    generated_x, generated_y = preprocess_univariate(generated_data, target_feature_index, n_lags=window)
    generated_X_train, generated_X_test, generated_y_train, generated_y_test = train_test_split(generated_x, generated_y, test_size=0.3, random_state=42)
    generated_train_dataset = TensorDataset(torch.Tensor(generated_X_train), torch.Tensor(generated_y_train).unsqueeze(1))
    generated_test_dataset = TensorDataset(torch.Tensor(generated_X_test), torch.Tensor(generated_y_test).unsqueeze(1))
    generated_test_loader = DataLoader(generated_test_dataset, batch_size=32, shuffle=False)
    generated_train_loader = DataLoader(generated_train_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.MSELoss()

    if os.path.exists(f'./predictive_measures/{model_name}/real_data_model.pth'):
        print("Model already exists")
    else:
        
        # Create an instance of the LSTM model
        model = LSTMModel(input_size=1, hidden_size=64, output_size=1)

        # Define the loss function and optimizer
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Train the model
        
        for epoch in range(5):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
        # Save the model
        if not os.path.exists(f'./predictive_measures/{model_name}'):
            os.makedirs(f'./predictive_measures/{model_name}')
        torch.save(model.state_dict(), f'./predictive_measures/{model_name}/real_data_model.pth')
    if os.path.exists(f'./predictive_measures/{model_name}/generated_data_model.pth'):
        print("Model already exists")
    else:
        # Create an instance of the LSTM model
        model = LSTMModel(input_size=1, hidden_size=64, output_size=1)

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # Train the model
        for epoch in range(5):
            model.train()
            for batch_X, batch_y in generated_train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
        # Save the model
        if not os.path.exists(f'./predictive_measures/{model_name}'):
            os.makedirs(f'./predictive_measures/{model_name}')
        torch.save(model.state_dict(), f'./predictive_measures/{model_name}/generated_data_model.pth')

    #Load the model
    real_data_model = LSTMModel(input_size=1, hidden_size=64, output_size=1)
    real_data_model.load_state_dict(torch.load(f'./predictive_measures/{model_name}/real_data_model.pth'))
    real_data_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = real_data_model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    average_loss_real = total_loss / len(test_loader)

    #Load the model
    generated_data_model = LSTMModel(input_size=1, hidden_size=64, output_size=1)
    generated_data_model.load_state_dict(torch.load(f'./predictive_measures/{model_name}/generated_data_model.pth'))
    generated_data_model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in generated_test_loader:
            outputs = generated_data_model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    average_loss_generated = total_loss / len(generated_test_loader)


    total_loss=0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = generated_data_model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    average_loss_realdata_generatedmodel = total_loss / len(test_loader)

    total_loss=0
    with torch.no_grad():
        for batch_X, batch_y in generated_test_loader:
            outputs = real_data_model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    average_loss_generateddata_realmodel = total_loss / len(generated_test_loader)
    dict={"Real Data Model":average_loss_real, "Generated Data Model":average_loss_generated, "Real Data Generated Model":average_loss_realdata_generatedmodel, "Generated Data Real Model":average_loss_generateddata_realmodel}
    df=pd.DataFrame(dict.items(), columns=["Model", "Loss"])
    df.to_csv(f"./predictive_measures/{model_name}/loss.csv", index=False)
    
    

    # Bar plot of the loss values
    plt.bar(df["Model"], df["Loss"])
    plt.xlabel("Model")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.savefig(f'./predictive_measures/{model_name}/loss.png')