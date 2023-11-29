import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_column(data, col_name):
    plt.figure(figsize=(10, 5))
    plt.plot(data[col_name], label=col_name)
    plt.xlabel('time')
    plt.title('Energy Consumed')
    plt.grid(True)
    plt.show()

def plot_output(y_pred_seq, y_true_seq, seq_len):
    pred_arr = []
    true_arr = []
    for i in range(0,len(y_pred_seq),seq_len):
        pred_arr.append(y_pred_seq[i])
        true_arr.append(y_true_seq[i])
    plt.figure(figsize=(10, 5))
    plt.plot(pred_arr, label='prediction_seq')
    plt.plot(true_arr, label='GndTruth_seq')
    plt.title('pred vs groundtruth')
    plt.xlabel('Seq index')
    plt.ylabel('Energy Consumed')
    plt.legend()
    plt.grid(True)
    plt.show()