import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Function to create sequences 
# output_seq_len <= input_seq_len
def create_sequences(input_data, output_data, input_seq_len, output_seq_len):
    input_seq = []
    output_seq = []
    for i in range(len(input_data) - input_seq_len - output_seq_len):
        input_seq.append(input_data[i:i+input_seq_len])
        output_seq.append(output_data[i+input_seq_len : i+input_seq_len+output_seq_len])
    return np.array(input_seq), np.array(output_seq)

def create_dataloaders(data, input_seq_len, output_seq_len, test_size, val_size, batch_size, rand_state):
    # Generate Input and Output Sequences
    input_seq, output_seq = create_sequences(data.values,data.values, input_seq_len, output_seq_len)

    # Split the data into training, validation, and test sets 
    X_train, X_test, y_train, y_test = train_test_split(input_seq, output_seq, test_size=test_size, random_state=rand_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=rand_state)  # 0.25 x 0.8 = 0.2

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # create dataloader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    val_data = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def parse_altitude(altitude_str):
    altitude_str = str(altitude_str)
    altitudes = [int(alt) for alt in altitude_str.split('-')]
    max_altitude = max(altitudes)
    min_altitude = min(altitudes)
    mean_altitude = sum(altitudes) / len(altitudes)
    return pd.Series([max_altitude, min_altitude, mean_altitude], index=['max_altitude', 'min_altitude', 'mean_altitude'])
    

def calculate_consumptions(dataset):

    # Calculate the time difference between each measurement
    dataset['time_diff'] = dataset.groupby('flight')['time'].diff().fillna(0)

    # Calculate current consumed in each interval (Coulombs = Amperes * Seconds)
    dataset['current_atm'] = dataset['battery_current'] * dataset['time_diff']

    # Calculate energy consumed in each interval (Coulombs = Amperes * Seconds)
    dataset['energy_atm'] = dataset['battery_current'] *dataset['battery_voltage']* dataset['time_diff']

    # Calculate the cumulative current consumption
    dataset['current_consumed'] = dataset.groupby('flight')['current_atm'].cumsum()

    # Calculate the cumulative energy consumption
    dataset['energy_consumed'] = dataset.groupby('flight')['energy_atm'].cumsum()
    return dataset

data = pd.read_csv('Quadcopter_dataset_cmu/flights.csv')    
print(data)

# Apply the function to the altitude column and join with the original dataframe
altitude_features = data['altitude'].apply(parse_altitude)
data = data.join(altitude_features)
# Now remove the original 'altitude' column as it's been replaced with numeric features
data.drop('altitude', axis=1, inplace=True)

#Convert Route ID to number 0-10
data['route'] = data['route'].replace(['R1','R2','R3','R4','R5','R6','R7','A1','A2','A3','H'],[0,1,2,3,4,5,6,7,8,9,10])
#Remove 'date' and 'time-day' from the dataset
data.drop(['date','time_day'], axis = 1, inplace = True)
#calculating power consumption (W)
data['power'] = data['battery_voltage'] * data['battery_current']

#calculation current consumption (Amp * s)
data = calculate_consumptions(data)


# Normalize the selected dataset features

features = ['wind_speed','wind_angle','battery_voltage', 'battery_current', 'position_x', 'position_y', 'position_z', 
                                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'velocity_x', 'velocity_y', 'velocity_z',
                                'angular_x', 'angular_y', 'angular_z','linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 
                                'speed', 'payload', 'max_altitude', 'min_altitude', 'mean_altitude','route','power','time_diff','current_atm',
                                'energy_atm','current_consumed','energy_consumed']

# Apply MinMaxScaler to the features except time & flight
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

print(data)


#Create Data loaders 

input_seq_len = 2
output_seq_len = 2
test_size = 0.2
val_size = 0.25
batch_size = 64
rand_state = 42

train_loader, val_loader, test_loader = create_dataloaders(data, input_seq_len, output_seq_len, test_size, val_size, batch_size, rand_state)
print(test_loader)