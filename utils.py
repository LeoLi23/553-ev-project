import numpy as np
import pandas as pd 

def parse_altitude(altitude_str):
    altitude_str = str(altitude_str)
    altitudes = [int(alt) for alt in altitude_str.split('-')]
    max_altitude = max(altitudes)
    min_altitude = min(altitudes)
    mean_altitude = sum(altitudes) / len(altitudes)
    return pd.Series([max_altitude, min_altitude, mean_altitude], index=['max_altitude', 'min_altitude', 'mean_altitude'])
    
# Function to create sequences 
def create_sequences(input_data, output_data, sequence_length):
    sequences = []
    output = []
    for i in range(len(input_data) - sequence_length):
        sequences.append(input_data[i:i+sequence_length])
        output.append(output_data[i+sequence_length])
    return np.array(sequences), np.array(output)