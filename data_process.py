import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.preprocessing import MinMaxScaler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Function to create sequences 
# output_seq_len <= input_seq_len
def create_sequences(input_data, output_data, input_seq_len, output_seq_len):
    input_seq = []
    output_seq = []
    for i in range(len(input_data) - input_seq_len):
        input_seq.append(input_data[i:i+input_seq_len])
        output_seq.append(output_data[i+input_seq_len : i+input_seq_len+output_seq_len])
    return np.array(input_seq), np.array(output_seq)


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

#Replace Altitude "25-50-100-25" to 50
data['altitude'] = data['altitude'].replace('25-50-100-25', 50)

#calculating power consumption (W)
data['power'] = data['battery_voltage'] * data['battery_current']

#calculation current consumption (Amp * s)
data = calculate_consumptions(data)

# print(data)

# Normalize the selected dataset features
features = ['wind_speed','wind_angle','battery_voltage', 'battery_current', 'position_x', 'position_y', 'position_z', 
                                'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'velocity_x', 'velocity_y', 'velocity_z',
                                'angular_x', 'angular_y', 'angular_z','linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 
                                'speed', 'payload', 'altitude','power','time_diff','current_atm','energy_atm','current_consumed','energy_consumed']


# Apply MinMaxScaler to the numeric features only
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
# print(data)

input_seq, output_seq = create_sequences(data.values,data.values, 2,2)

print("input_seq")
print(input_seq[0:4])
print("output_seq")
print(output_seq[0:4])

