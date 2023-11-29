import matplotlib.pyplot as plt

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

