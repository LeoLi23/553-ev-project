import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from data_process import get_data_loaders

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Function to create sequences
def create_sequences(input_data, output_data, input_seq_len, output_seq_len, future_cov_len):
    input_seq = []
    output_seq = []
    future_covariates_seq = []
    
    total_len = input_seq_len + output_seq_len + future_cov_len
    
    for i in range(len(input_data) - total_len + 1):
        input_seq.append(input_data[i:(i + input_seq_len)])
        output_seq.append(output_data[i + input_seq_len:(i + input_seq_len + output_seq_len)])
        future_covariates_seq.append(input_data[i + input_seq_len:(i + input_seq_len + future_cov_len)])
    
    return np.array(input_seq), np.array(output_seq), np.array(future_covariates_seq)

def create_dataloaders(data, input_seq_len, output_seq_len, future_cov_len, test_size, val_size, batch_size, rand_state):
    # Assuming data is already preprocessed and scaled
    input_features = getFeatures()  # Add more features if necessary
    output_feature = 'energy_consumed'
    
    # Generate Input, Output, and Future Covariates Sequences
    input_seq, output_seq, future_cov_seq = create_sequences(
        data[input_features].values,
        data[output_feature].values,
        input_seq_len,
        output_seq_len,
        future_cov_len
    )
    
    X_train, X_test, y_train, y_test, future_cov_train, future_cov_test = train_test_split(input_seq, output_seq, future_cov_seq, test_size=test_size, random_state=rand_state)
    X_train, X_val, y_train, y_val, future_cov_train, future_cov_val = train_test_split(X_train, y_train, future_cov_train, test_size=val_size, random_state=rand_state)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    future_cov_train = torch.tensor(future_cov_train, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    future_cov_val = torch.tensor(future_cov_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    future_cov_test = torch.tensor(future_cov_test, dtype=torch.float32)

    # Create DataLoader
    train_data = TensorDataset(X_train, y_train, future_cov_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    val_data = TensorDataset(X_val, y_val, future_cov_val)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    test_data = TensorDataset(X_test, y_test, future_cov_test)
    test_loader = DataLoader(test_data, batch_size=batch_size) 

    return data, train_loader, val_loader, test_loader

def getFeatures():
    return ['wind_speed','wind_angle','battery_voltage', 'battery_current', 'position_x', 'position_y', 'position_z', 
                                    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'velocity_x', 'velocity_y', 'velocity_z',
                                    'angular_x', 'angular_y', 'angular_z','linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 
                                    'speed', 'payload', 'max_altitude', 'min_altitude', 'mean_altitude','route','power','time_diff','current_atm',
                                    'energy_atm','current_consumed','energy_consumed']
    
def get_data_loaders_TCN():
    # Example usage
    sequence_length = 10  # Define the sequence length for input
    forecast_horizon = 2  # Define the forecast horizon for output
    future_cov_len = 3  # Define the length of future covariates
    data = get_data_loaders()[0]  # Load your data
    
    return create_dataloaders(
        data,
        input_seq_len=sequence_length,
        output_seq_len=forecast_horizon,
        future_cov_len=future_cov_len,
        test_size=0.2,
        val_size=0.25,
        batch_size=64,
        rand_state=42
    )
