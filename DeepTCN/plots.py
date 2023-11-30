import matplotlib.pyplot as plt

def plot_matplotlib(df, quantiles, x_axis_label = "time", y_axis_label = "values"):
    """
    Plot the target time series and the predicted quantiles using matplotlib.

    Parameters:
    __________________________________
    df: pd.DataFrame.
        Data frame with target time series and predicted quantiles.
    quantiles: list.
        Quantiles of target time series which have been predicted.
    x_axis_label: str.
        Label for the x-axis.
    y_axis_label: str.
        Label for the y-axis.

    Returns:
    __________________________________
    None. Displays the matplotlib plot.
    """
    
    # get the number of predicted quantiles
    n_quantiles = len(quantiles)

    # get the number of targets
    n_targets = int((df.shape[1] - 1) / (n_quantiles + 1))

    # Create subplots
    fig, axs = plt.subplots(n_targets, 1, figsize=(10, 5 * n_targets), sharex=True)

    # If only one target, axs is not a list, so we convert it to a list for consistent handling
    if n_targets == 1:
        axs = [axs]

    for i in range(n_targets):
        # Plot the actual values
        axs[i].plot(df['time_idx'], df['target_' + str(i + 1)], label='Actual', color='#afb8c1')

        # Plot the median
        axs[i].plot(df['time_idx'], df['target_' + str(i + 1) + '_0.5'], label='Median', color='blue')

        # Plot the quantile ranges
        for j in range(n_quantiles // 2):
            axs[i].fill_between(
                df['time_idx'], 
                df['target_' + str(i + 1) + '_' + str(quantiles[j])],
                df['target_' + str(i + 1) + '_' + str(quantiles[-(j + 1)])],
                color='blue', 
                alpha=0.1 * (j + 1),
                label='q{} - q{}'.format(quantiles[j], quantiles[-(j + 1)])
            )

        axs[i].set_title('Target ' + str(i + 1))
        axs[i].set_xlabel(x_axis_label)
        axs[i].set_ylabel(y_axis_label)

        # Only show legend on the first subplot for clarity
        if i == 0:
            axs[i].legend(loc='upper left', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


