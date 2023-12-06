import torch
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def getFeatures():
    return ['wind_speed','wind_angle','battery_voltage', 'battery_current', 'position_x', 'position_y', 'position_z', 
                                    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'velocity_x', 'velocity_y', 'velocity_z',
                                    'angular_x', 'angular_y', 'angular_z','linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 
                                    'speed', 'payload', 'max_altitude', 'min_altitude', 'mean_altitude','route','power','time_diff','current_atm',
                                    'energy_atm','current_consumed','energy_consumed', 'x_future', 'y_future', 'z_future']
    

# Function to create sequences 
# output_seq_len <= input_seq_len
def create_sequences_new(data, target, input_seq_len, output_seq_len, shuffle = 0):
    input_seq = []
    output_seq = []
    features = getFeatures()
    flight_ids = data['flight'].unique()

    for id in flight_ids:
        flight_input_data = data[data['flight'] == id][features].values
        flight_output_data = data[data['flight'] == id][target].values
        for i in range(len(flight_input_data) - input_seq_len - output_seq_len):
            input_seq.append(flight_input_data[i:i+input_seq_len])
            output_seq.append(flight_output_data[i+input_seq_len : i+input_seq_len+output_seq_len])
    
    if shuffle == 1:
        random.shuffle(input_seq)
        random.shuffle(output_seq)
        
    return np.array(input_seq), np.array(output_seq)

def create_sequences(input_data, output_data, input_seq_len, output_seq_len):
    input_seq = []
    output_seq = []
    for i in range(len(input_data) - input_seq_len - output_seq_len):
        input_seq.append(input_data[i:i+input_seq_len])
        output_seq.append(output_data[i+input_seq_len : i+input_seq_len+output_seq_len])
    return np.array(input_seq), np.array(output_seq)

def create_dataloaders(data, input_seq_len, output_seq_len, test_size, val_size, batch_size, rand_state, 
                       target:str, features:list=None, covariates:bool=False):
    # Generate Input and Output Sequences
    features = getFeatures()
    input_seq, output_seq = create_sequences(data[features].values,data[target].values, input_seq_len, output_seq_len)

    # print(data["energy_consumed"].values)
    
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

def create_dataloaders_by_flights(data, input_seq_len, output_seq_len, test_size, val_size, batch_size, rand_state, 
                                  target: str, features: list=None, covariates: bool=False):
    ##
    # Getting data sets by unique flights
    uniq_flights = data['flight'].unique().tolist()
    num_flights = len(uniq_flights)
    
    # By paper, [268, 269, 270, 271, 272, 273, 276, 277, 278, 279] are random flights and included in test set
    force_test_flights = [268, 269, 270, 271, 272, 273, 276, 277, 278, 279]
    uniq_flights = list(set(uniq_flights) - set(force_test_flights))

    # Split the flights into training, validation, and test sets
    random.Random(rand_state).shuffle(uniq_flights)
    num_other_tests = max(0, round(num_flights * test_size) - 10)
    test_flights = uniq_flights[:num_other_tests] + [268, 269, 270, 271, 272, 273, 276, 277, 278, 279]
    
    num_val_flights = round(num_flights * val_size)
    val_flights = uniq_flights[num_other_tests:num_other_tests + num_val_flights]

    train_flights = uniq_flights[num_other_tests + num_val_flights:]

    # # Split the data into training, validation, and test sets 
    # train_flights, test_flights = train_test_split(uniq_flights, test_size= test_size, random_state=rand_state)
    # train_flights, val_flights = train_test_split(train_flights, test_size = val_size, random_state=rand_state)

    train_data = data[data['flight'].isin(train_flights)]
    val_data = data[data['flight'].isin(val_flights)]
    test_data = data[data['flight'].isin(test_flights)]
    # print("TRAIN DATA")
    # print(train_data)

    # print("VAL DATA")
    # print(val_data)

    # print("TEST DATA")
    # print(test_data)
    # Generate Input and Output Sequences
    features = getFeatures()
    train_input_seq, train_output_seq = create_sequences(train_data[features].values,train_data[target].values, input_seq_len, output_seq_len)
    val_input_seq, val_output_seq = create_sequences(val_data[features].values,val_data[target].values, input_seq_len, output_seq_len)
    test_input_seq, test_output_seq = create_sequences(test_data[features].values,test_data[target].values, input_seq_len, output_seq_len)

    # Convert to PyTorch tensors
    X_train = torch.tensor(train_input_seq, dtype=torch.float32)
    y_train = torch.tensor(train_output_seq, dtype=torch.float32)
    X_val = torch.tensor(val_input_seq, dtype=torch.float32)
    y_val = torch.tensor(val_output_seq, dtype=torch.float32)
    X_test = torch.tensor(test_input_seq, dtype=torch.float32)
    y_test = torch.tensor(test_output_seq, dtype=torch.float32)

    # create dataloader
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    val_data = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, val_loader, test_loader, train_flights, val_flights, test_flights
    ##



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


def calculate_futures(dataset, step_into_future = 12):
    dataset['x_future'] = dataset.groupby('flight')['position_x'].shift(-step_into_future)
    dataset['y_future'] = dataset.groupby('flight')['position_y'].shift(-step_into_future)
    dataset['z_future'] = dataset.groupby('flight')['position_z'].shift(-step_into_future)
    dataset['x_change'] = dataset.groupby('flight')['position_x'].diff()
    dataset['y_change'] = dataset.groupby('flight')['position_y'].diff()
    dataset['z_change'] = dataset.groupby('flight')['position_z'].diff()
    dataset = dataset.dropna()
    return dataset 


def get_data_loaders(data, input_seq_len = 10, output_seq_len = 2,
    test_size = 0.2,
    val_size = 0.25,
    batch_size = 64,
    rand_state = 42,
    target = 'power',
    trim:float=None,
    features:list=None,
    covariates:bool=False):
    # Read the data from the CSV file
    # data = pd.read_csv('flights.csv')    

    #Drop broken data records for flight 211 -219
    flights_to_remove = [211,212,213,214,215,216,217,218,219]
    data = data[~data['flight'].isin(flights_to_remove)]
    
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
    if covariates:
        data = calculate_futures(data, output_seq_len)

    # TODO: add air density as a feature (refer to paper code)

    if features is None:
        features = getFeatures()
    if covariates:
        features = features + ['x_future','y_future','z_future'] + ['x_change', 'y_change','z_change']

    # Apply MinMaxScaler to the features except time & flight
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])

    #Create Data loaders 

    train_loader, val_loader, test_loader, train_flights, val_flights, test_flights = create_dataloaders_by_flights(data, input_seq_len, output_seq_len, test_size, val_size, batch_size, rand_state, target)
    return data, train_loader, val_loader, test_loader, train_flights, val_flights, test_flights

def getFeatures():
    return ['wind_speed','wind_angle','battery_voltage', 'battery_current', 'position_x', 'position_y', 'position_z', 
                                    'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'velocity_x', 'velocity_y', 'velocity_z',
                                    'angular_x', 'angular_y', 'angular_z','linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 
                                    'speed', 'payload', 'max_altitude', 'min_altitude', 'mean_altitude','route','power','time_diff','current_atm',
                                    'energy_atm','current_consumed','energy_consumed', 
                                    'x_future', 'y_future', 'z_future', 'x_change', 'y_change','z_change']
    

if __name__ == "__main__":
    data = pd.read_csv('flights.csv') 
    # data, train_loader, val_loader, test_loader = get_data_loaders(data, 24, 10)
    features = getFeatures()
    print(len(getFeatures()))

    data, train_loader, val_loader, test_loader, train_flights, val_flights, test_flights = get_data_loaders(data, 24,10, covariates=True)
    flight = random.choice(test_flights)
    print(flight)
    print(data['x_change'])

    #testing create_sequences_new()
    input1,output1 = create_sequences(data[features].values, data['power'].values, 24,10)
    print("Sequences dimension not separated by flight")
    print(input1.shape)
    print(output1.shape)

    input2,output2 = create_sequences_new(data, 'power', 24,10)
    print("Sequences dimensions separated by flights")
    print(input2.shape)
    print(output2.shape)


""""
TODO: 
1) add air density
2) add location difference -- Done
3) create seq strictly based on flight class -- DONE, create_sequences_new()
4) create function to create seq based on flights -- Done, created unique lists for test/train flights
5) 
"""