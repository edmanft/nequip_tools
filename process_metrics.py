import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import plotly.graph_objects as go

def load_filtered_metrics(folder_path):
    epoch_file_path = os.path.join(folder_path, 'filter_metrics_epoch.csv')
    initialization_file_path = os.path.join(folder_path, 'filter_metrics_initialization.csv')
    
    # Check if the epoch file exists
    if not os.path.exists(epoch_file_path):
        print(f"Error: File not found {epoch_file_path}")
        return None, None

    # Load the two specified CSV files
    epoch_data = pd.read_csv(epoch_file_path)
    initialization_data = pd.read_csv(initialization_file_path)
    
    return epoch_data, initialization_data

def process_wall_times(epoch_data):
    # Verify that 'wall' column exists
    if ' wall' not in epoch_data.columns:
        print("Error: 'wall' column not found in the dataset.")
        return None, None, None, None
    
    # Extract 'wall' column
    wall_times = np.array(list(epoch_data[' wall']))
    time_per_epoch = wall_times.copy()
    # Compute time per epoch by subtracting the previous time from the current time
    time_per_epoch[1:] = time_per_epoch[1:] - time_per_epoch[:-1]  
    # Compute statistics
    mean_wall = time_per_epoch.mean()
    std_wall = time_per_epoch.std()
    min_wall = time_per_epoch.min()
    max_wall = time_per_epoch.max()
    
    return mean_wall, std_wall, min_wall, max_wall

def plot_losses(epoch_data, initialization_data, args):
    # Extract the relevant metrics
    metrics = ['training_loss_f', 'training_loss_e', 'training_loss', 
               'validation_loss_f', 'validation_loss_e', 'validation_loss']
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and styles
    colors = {'training': 'green', 'validation': 'orange', 'initialization': 'red'}
    styles = {'f': '-', 'e': '--', 'loss': ':'}  # Empty for generic 'training_loss' and 'validation_loss'
    plotly_styles = {'-': 'solid', '--': 'dash', ':': 'dot'}

    # Plot training and validation metrics
    for metric in metrics:
        if metric in epoch_data.columns:
            if 'training' in metric:
                ax.plot(epoch_data['epoch'], epoch_data[metric], color=colors['training'], linestyle=styles[metric.split('_')[-1]], label=metric)
            elif 'validation' in metric:
                ax.plot(epoch_data['epoch'], epoch_data[metric], color=colors['validation'], linestyle=styles[metric.split('_')[-1]], label=metric)
    
    # Plot initialization metrics (horizontal lines)
    for metric in ['validation_loss_f', 'validation_loss_e', 'validation_loss']:
        if metric in initialization_data.columns:
            ax.axhline(y=initialization_data[metric].values[0], color=colors['initialization'], linestyle=styles[metric.split('_')[-1]], label=f'init_{metric}')
    
    # Labels, legend, and layout
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.legend()
    plt.title('Train/val RMSE loss')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot as PDF
    plt.savefig(os.path.join(args.path, 'training_validation_losses.pdf'), format='pdf')
    plt.show()

    # Plot with Plotly
    fig_plotly = go.Figure()

    # Add traces for training and validation metrics
    for metric in metrics:
        if metric in epoch_data.columns:
            if 'training' in metric:
                fig_plotly.add_trace(go.Scatter(x=epoch_data['epoch'], y=epoch_data[metric],
                                                mode='lines', name=metric,
                                                line=dict(color=colors['training'], dash=plotly_styles[styles[metric.split('_')[-1]]])))
            elif 'validation' in metric:
                fig_plotly.add_trace(go.Scatter(x=epoch_data['epoch'], y=epoch_data[metric],
                                                mode='lines', name=metric,
                                                line=dict(color=colors['validation'], dash=plotly_styles[styles[metric.split('_')[-1]]])))
    
    # Add traces for initialization metrics
    for metric in ['validation_loss_f', 'validation_loss_e', 'validation_loss']:
        if metric in initialization_data.columns:
            fig_plotly.add_trace(go.Scatter(x=[0, max(epoch_data['epoch'])],
                                            y=[initialization_data[metric].values[0]]*2,
                                            mode='lines', name=f'init_{metric}',
                                            line=dict(color=colors['initialization'], dash=plotly_styles[styles[metric.split('_')[-1]]])))
    
    # Update layout
    fig_plotly.update_layout(title='Train/val RMSE loss',
                             xaxis_title='Epoch',
                             yaxis_title='RMSE',
                             legend_title='Metrics')
    
    # Save the plot as HTML
    fig_plotly.write_html(os.path.join(args.path, 'training_validation_losses.html'))

def find_best_epoch_and_print_metrics(epoch_data):
    # Find the epoch with the best (minimum) validation loss
    best_epoch_index = epoch_data['validation_loss'].idxmin()
    best_epoch_data = epoch_data.loc[best_epoch_index]
    
    # Extract and print relevant metrics
    print(f"Best Epoch: {int(best_epoch_data['epoch'])}")
    print(f"Training Metrics at Best Epoch:")
    print(f"  training_loss_f: {best_epoch_data['training_loss_f']}")
    print(f"  training_loss_e: {best_epoch_data['training_loss_e']}")
    print(f"  training_loss: {best_epoch_data['training_loss']}")
    print(f"  training_f_mae: {best_epoch_data['training_f_mae']}")
    print(f"  training_e_mae: {best_epoch_data['training_e_mae']}")
    print(f"Validation Metrics at Best Epoch:")
    print(f"  validation_loss_f: {best_epoch_data['validation_loss_f']}")
    print(f"  validation_loss_e: {best_epoch_data['validation_loss_e']}")
    print(f"  validation_loss: {best_epoch_data['validation_loss']}")
    print(f"  validation_f_mae: {best_epoch_data['validation_f_mae']}")
    print(f"  validation_e_mae: {best_epoch_data['validation_e_mae']}")




def main():
    parser = argparse.ArgumentParser(description='Process CSV files to calculate epoch times from cumulative wall times.')
    parser.add_argument('-p', '--path', required=True, help='Directory path containing the metrics CSV files.')
    args = parser.parse_args()
    args.path = os.path.abspath(args.path)
    epoch_data, initialization_data = load_filtered_metrics(args.path)
    mean_wall, std_wall, min_wall, max_wall = process_wall_times(epoch_data)
    print(f"Wall time per epoch (s): \nmean: {mean_wall}   std: {std_wall}    min: {min_wall}  max: {max_wall}")
    find_best_epoch_and_print_metrics(epoch_data)
    plot_losses(epoch_data, initialization_data, args)


if __name__ == "__main__":
    main()
