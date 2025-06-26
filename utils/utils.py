# Helper functions (metrics, visualizations, etc.)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import datetime
import pyfiglet
from prettytable import PrettyTable
import time
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torchinfo import summary
from typing import Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def project_intro(name):
    ascii_art = pyfiglet.figlet_format(name,font="big")
    print(ascii_art)

def model_summary(model, input_size, **kwargs):
    print("Model Details as follows: ")
    # Get summary data
    info = summary(model, input_size=input_size, verbose=0, **kwargs)

    # Left: Model layers table
    model_table = PrettyTable()
    model_table.field_names = ["Layer (type)", "Output Shape", "Param #"]
    for layer in info.summary_list:
        model_table.add_row([
            layer.class_name,
            str(layer.output_size),
            f"{layer.num_params:,}"
        ])
    model_lines = model_table.get_string().splitlines()

    # Right: Stats table
    stats_table = PrettyTable()
    stats_table.field_names = ["Stats", "Count"]
    stats_table.add_row(["Total", f"{info.total_params:,}"])
    stats_table.add_row(["Trainable", f"{info.trainable_params:,}"])
    stats_table.add_row(["Non-trainable", f"{info.total_params - info.trainable_params:,}"])
    stats_table.add_row(["Total mult-adds", f"{info.total_mult_adds:,}"])
    stats_lines = stats_table.get_string().splitlines()

    # Pad the shorter table
    max_len = max(len(model_lines), len(stats_lines))
    model_lines += [" " * len(model_lines[0])] * (max_len - len(model_lines))
    stats_lines += [" " * len(stats_lines[0])] * (max_len - len(stats_lines))

    # Combine line by line
    for m_line, s_line in zip(model_lines, stats_lines):
        print(f"{m_line}   {s_line}")


def training_knobs(cfg_dict):
    print("Training Knobs:  ")
    items = list(cfg_dict.items())
    mid = len(items) // 2 + len(items) % 2  # Left half gets extra if odd

    left_items = items[:mid]
    right_items = items[mid:]

    # Prepare row strings
    left_table = PrettyTable()
    left_table.field_names = ["Parameter", "Value"]
    for k, v in left_items:
        left_table.add_row([k, v])
    left_lines = left_table.get_string().splitlines()

    right_table = PrettyTable()
    right_table.field_names = ["Parameter", "Value"]
    for k, v in right_items:
        right_table.add_row([k, v])
    right_lines = right_table.get_string().splitlines()

    # Pad shorter table if needed
    max_lines = max(len(left_lines), len(right_lines))
    left_lines += [" " * len(left_lines[0])] * (max_lines - len(left_lines))
    right_lines += [" " * len(right_lines[0])] * (max_lines - len(right_lines))

    # Print side by side
    for l, r in zip(left_lines, right_lines):
        print(f"{l}   {r}")


def compute_metrics(y_true, y_pred):
    """Compute common classification metrics and return them as a dictionary.
    
    Calculates accuracy, precision, recall, and F1 score using scikit-learn metrics.
    For precision, recall, and F1, macro averaging is used to handle multi-class classification.
    
    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        
    Returns:
        dict: Dictionary containing the following metrics:
            - accuracy (float): Accuracy score
            - precision (float): Macro-averaged precision score
            - recall (float): Macro-averaged recall score
            - f1_score (float): Macro-averaged F1 score
            
    Example:
        >>> y_true = [0, 1, 0, 1]
        >>> y_pred = [0, 0, 0, 1]
        >>> compute_metrics(y_true, y_pred)
        {
            'accuracy': 0.75,
            'precision': 0.75,
            'recall': 0.75,
            'f1_score': 0.6666666666666666
        }
        
    Note:
        - Uses 'macro' averaging for multi-class metrics (treats all classes equally)
        - Sets zero_division=0 to handle cases with no predicted samples
    """

    return (accuracy_score(y_true, y_pred),precision_score(y_true, y_pred, average='weighted', zero_division=0),recall_score(y_true, y_pred, average='weighted', zero_division=0),f1_score(y_true, y_pred, average='weighted', zero_division=0)) 

    # return {
    #     "accuracy": accuracy_score(y_true, y_pred),
    #     "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
    #     "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
    #     "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    # }


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Save a model's state dictionary to the specified file path.
    
    Preserves the model's learned parameters for later reuse or continued training.
    Uses PyTorch's native serialization method for efficient storage.
    
    Args:
        model (torch.nn.Module): PyTorch model instance to be saved
        path (str | Path): File path where the model weights will be stored.
            Should have .pt or .pth file extension
            
    Returns:
        None: The model weights are written to disk but nothing is returned
        
    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> save_checkpoint(model, "trained_model.pth")
        >>> loaded_model = torch.nn.Linear(10, 2)
        >>> loaded_model.load_state_dict(torch.load("trained_model.pth"))
        
    Notes:
        - Creates a full snapshot of model parameters (weights and biases)
        - Does not save optimizer state or training progress - only model weights
        - Ensure directory structure exists before saving
        - File can be loaded with torch.load() and applied to a compatible architecture
        - For multi-GPU models, considers model wrapped in DataParallel/MultiGPU wrappers
    """
    torch.save(model.state_dict(), path)



def save_at_n_epoch(
    model: torch.nn.Module,
    epoch: int,
    path: Union[str, Path],
    config: object
) -> str:
    """Conditionally save model checkpoint at specified epoch intervals and final epoch.
    
    Checks if either:
    1. Current epoch is a multiple of `save_every_n_epochs` (when > 0), or
    2. Current epoch is the final epoch specified in config
    Saves model weights if either condition is met.

    Args:
        model: PyTorch model instance to save
        epoch: Current epoch number (1-based or 0-based depending on training loop)
        path: Destination path for model checkpoint. Recommended extensions: .pt, .pth
        config: Configuration object with attributes:
            - save_every_n_epochs (int): Save interval (0 disables intermediate saves)
            - num_epochs (int): Total number of training epochs

    Returns:
        str: Informational message if saved, empty string otherwise. Message contains
            a leading newline for cleaner output formatting in training logs.

    Example:
        >>> class Config:
        ...     save_every_n_epochs = 2
        ...     num_epochs = 5
        >>> config = Config()
        >>> save_at_n_epoch(model, 2, "model.pth", config)
        '\nThe model is saved at 2 epoch'
        >>> save_at_n_epoch(model, 5, "model.pth", config)
        '\nThe model is saved at 5 epoch'
        >>> save_at_n_epoch(model, 3, "model.pth", config)
        ''

    Notes:
        - Uses save_checkpoint() for actual saving (only saves model state_dict)
        - Always saves at final epoch regardless of save_every_n_epochs value
        - Returns message with leading newline to separate from progress bars
        - Ensure parent directories exist before calling
        - Set config.save_every_n_epochs=0 to only save final model
        - Epoch numbering should match training loop's actual epoch count
    """
    if (config.save_every_n_epochs > 0 and (epoch) % config.save_every_n_epochs == 0) or epoch == config.training_config["epochs"]:
        save_checkpoint(model, path)
        return f'\nThe model is saved at {epoch} epoch'
    return ''


def get_date() -> str:
    """Get current date and time in standardized format.
    
    Returns:
        str: Current local date/time formatted as ISO 8601 string (YYYY-MM-DD HH:MM:SS)
        
    Example:
        >>> get_date()
        '2023-08-05 15:30:00'
        
    Notes:
        - Uses local system time and datetime
        - Consistent 24-hour time format
        - Fixed length format (19 characters) for reliable parsing
        - Suitable for timestamps in logging or file naming
    """
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_time_diff(start: float, end: float) -> str:
    """Calculate and format time difference between two timestamps.
    
    Converts the absolute difference between two timestamps into a human-readable
    string using appropriate units (seconds, minutes, hours). Automatically selects
    the largest suitable unit for representation.

    Args:
        start: Start timestamp in seconds (e.g., from time.time())
        end: End timestamp in seconds (e.g., from time.time())

    Returns:
        str: Formatted time difference with unit, rounded to 3 decimal places.
            Possible formats:
            - "X.XXX seconds" (for differences < 60 seconds)
            - "X.XXX minutes" (for differences < 1 hour)
            - "X.XXX hours" (for differences >= 1 hour)

    Example:
        >>> get_time_diff(1622505600, 1622505600.123)
        '0.123 seconds'
        >>> get_time_diff(1622505600, 1622505660)  # 60 seconds difference
        '1.0 minutes'
        >>> get_time_diff(1622505600, 1622509200)  # 3600 seconds (1 hour)
        '1.0 hours'
        >>> get_time_diff(1622505600, 1622505600)  # Zero difference
        '0.0 seconds'

    Notes:
        - Always returns absolute difference (order of timestamps doesn't matter)
        - Uses 3 decimal places for all measurements
        - Units are always pluralized ("1.0 minutes" instead of "1.0 minute")
        - Thresholds: 
            60 seconds = 1 minute boundary
            3600 seconds = 1 hour boundary
    """
    interval = abs(end - start)  # ensure positive interval
        
    if interval < 60:
        return f"{round(interval, 3)} seconds"
    elif interval < 3600:
        minutes = round(interval / 60, 3)
        return f"{minutes} minutes"
    else:
        hours = round(interval / 3600, 3)
        return f"{hours} hours"



def print_dist(*data, config):
    print('Class Distribution as follows: ')
    with open(config.result_dir/'data_distribution.txt', 'w') as f:
        f.write('\nClass Distribution as follows:\n')
    table = PrettyTable()
    table.field_names = ["Class", "Train", "Test"]
    for cls in ['CN', 'MCI', 'AD']:
        table.add_row([cls, data[0][cls], data[1][cls]])

    print(table)

    os.makedirs(config.result_dir, exist_ok= True)
    with open(config.result_dir/'data_distribution.txt', 'a') as f:
        f.write(str(table))


def plot_single_metrics(config):
    # Load metrics from JSON
    metric_path = os.path.join(config.metrics_dir, 'metrics.json')
    with open(metric_path, 'r') as f:
        metrics = json.load(f)

    figure_dir = config.figures_dir
    os.makedirs(figure_dir, exist_ok=True)

    # Extract the metrics
    train_loss = metrics["train_loss"]
    test_loss = metrics["test_loss"]
    train_accuracy = metrics["train_accuracy"]
    test_accuracy = metrics["test_accuracy"]
    train_precision = metrics["train_precision"]
    test_precision = metrics["test_precision"]
    train_recall = metrics["train_recall"]
    test_recall = metrics["test_recall"]
    train_f1 = metrics["train_f1_score"]
    test_f1 = metrics["test_f1_score"]

    # Plotting
    fig, axs = plt.subplots(5, 1, figsize=(12, 24), sharex=True)

    # Loss plot
    axs[0].plot(train_loss, label='Train Loss')
    axs[0].plot(test_loss, label='Test Loss')
    axs[0].set_title('(a). Loss over Epochs')
    axs[0].legend()

    # Accuracy plot
    axs[1].plot(train_accuracy, label='Train Accuracy')
    axs[1].plot(test_accuracy, label='Test Accuracy')
    axs[1].set_title('(b). Accuracy over Epochs')
    axs[1].legend()

    # Recall plot
    axs[2].plot(train_recall, label='Train Recall')
    axs[2].plot(test_recall, label='Test Recall')
    axs[2].set_title('(c). Recall over Epochs')
    axs[2].legend()

    # Precision plot
    axs[3].plot(train_precision, label='Train Precision')
    axs[3].plot(test_precision, label='Test Precision')
    axs[3].set_title('(d). Precision over Epochs')
    axs[3].legend()

    # F1-score plot
    axs[4].plot(train_f1, label='Train F1-score')
    axs[4].plot(test_f1, label='Test F1-score')
    axs[4].set_title('(e). F1-score over Epochs')
    axs[4].legend()

    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'metrics.png'))
    plt.close()


def plot_metrics(config):
    # Load metrics from JSON
    metric_path = os.path.join(config.metrics_dir, 'metrics.json')
    with open(metric_path, 'r') as f:
        metrics = json.load(f)

    # Create DataFrame
    epochs = list(range(1, len(metrics['train_loss']) + 1))
    train_df = pd.DataFrame({
        'epoch': epochs,
        'loss': metrics['train_loss'],
        'accuracy': metrics['train_accuracy'],
        'precision': metrics['train_precision'],
        'recall': metrics['train_recall'],
        'f1_score': metrics['train_f1_score']
    })

    test_df = pd.DataFrame({
        'epoch': epochs,
        'loss': metrics['test_loss'],
        'accuracy': metrics['test_accuracy'],
        'precision': metrics['test_precision'],
        'recall': metrics['test_recall'],
        'f1_score': metrics['test_f1_score']
    })

    # Melt DataFrames for easier Seaborn plotting
    train_melted = train_df.melt(id_vars='epoch', var_name='metric', value_name='scores')
    test_melted = test_df.melt(id_vars='epoch', var_name='metric', value_name='scores')

    figure_dir_train = os.path.join(config.figures_dir, 'train')
    os.makedirs(figure_dir_train, exist_ok=True)

    # Plot training metrics
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=train_melted, x='epoch', y='scores', hue='metric')
    plt.title('(a). Training Metrics vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir_train, 'assessments.png'))
    plt.close()

    figure_dir_test = os.path.join(config.figures_dir, 'test')
    os.makedirs(figure_dir_test, exist_ok=True)
    # Plot testing metrics
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=test_melted, x='epoch', y='scores', hue='metric')
    plt.title('(b). Testing Metrics vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir_test, 'assessments.png'))
    plt.close()


def plot_cm(y_true, y_pred, config, epoch, phase = 'Train'):
    # confusion matrix plot

    cm = confusion_matrix(y_true, y_pred)
    labels = ['CN','MCI','AD']

    figure_dir = os.path.join(config.figures_dir,f'{phase.lower()}')
    os.makedirs(figure_dir, exist_ok=True)
    
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels = labels)
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title(f"{phase}-Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir,f'confusion_matrix_{epoch+1}.png'))
    plt.close()

def plot_save_config(config):
    def convert(v):
        if isinstance(v, Path):
            return str(v)
        elif isinstance(v, torch.device):
            return str(v)
        else:
            return v

    # Convert all values in the config dictionary
    converted_dict = {k: convert(v) for k, v in config.__dict__.items()}

    obj_dict = {
        "configs": converted_dict
    }

    # Optional: Check types
    # check = {k: [v, type(v)] for k, v in converted_dict.items()}
    # print("Type Check:", check)

    # Save to JSON
    config_path = os.path.join(config.metrics_dir, 'configs.json')
    os.makedirs(config.metrics_dir, exist_ok=True)  # Ensure directory exists

    with open(config_path, 'w') as f:
        json.dump(obj_dict, f, indent=4)

from collections import defaultdict

def data_dist(dataset):
    id_to_label = {0: 'CN', 1: 'MCI', 2: 'AD'}
    label_dist = defaultdict(int)
    
    for i in dataset:
        label = i['label']
        label_idx = torch.argmax(label).item()
        label_dist[id_to_label[label_idx]] += 1
    
    return dict(label_dist)


# def check_shapes(dictionary):
#     for key, value in dictionary.items():
#         # Check if the value has a shape attribute (like numpy arrays or pandas DataFrames)
#         if hasattr(value, 'shape'):
#             print(f"Key: {key}, Shape: {value.shape}")
#         # If the value is a list, check its length and the length of inner lists if nested
#         elif isinstance(value, list):
#             try:
#                 print(f"Key: {key}, Shape: ({len(value)}, {len(value[0]) if isinstance(value[0], list) else 'n/a'})")
#             except IndexError:
#                 print(f"Key: {key}, Shape: ({len(value)},)")
#         # For anything else, just print the type
#         else:
#             print(f"Key: {key}, Type: {type(value)}")


# def pad_sequence(sequences):
#     max_size = max([s.size() for s in sequences])
#     padded_sequences = [F.pad(s, (0, max_size[2] - s.size(2), 0, max_size[1] - s.size(1), 0, max_size[0] - s.size(0))) for s in sequences]
#     return torch.stack(padded_sequences)

# def my_collate_fn(batch):
#     batch = {k: [d[k] for d in batch] for k in batch[0]}
#     for k in batch:
#         if k == 'image_data':
#             batch[k] = pad_sequence(batch[k])
#         elif k == 'atlas_data':
#             atlas_data = {atlas: [] for atlas in batch[k][0].keys()}
#             for sample in batch[k]:
#                 for atlas, data in sample.items():
#                     atlas_data[atlas].append(data)
#             for atlas in atlas_data:
#                 atlas_data[atlas] = default_collate(atlas_data[atlas])
#             batch[k] = atlas_data
#         else:
#             batch[k] = default_collate(batch[k])
#     return batch


# def train_test_split(dataset, train_ratio=0.7):
#     train_size = int(len(dataset) * train_ratio)
#     test_size = len(dataset) - train_size
#     return random_split(dataset, [train_size, test_size])


# def visualize_attention_weights(attention_weights, title, save_path=None):
#     """
#     Safely visualize attention weights with proper reshaping
    
#     Args:
#         attention_weights: torch.Tensor or numpy array
#         title: str, title for the plot
#         save_path: str, optional path to save the figure
#     """
#     if isinstance(attention_weights, torch.Tensor):
#         attention_weights = attention_weights.detach().cpu().numpy()
    
#     # Ensure 2D shape
#     if len(attention_weights.shape) == 1:
#         attention_weights = attention_weights.reshape(-1, 1)
#     elif len(attention_weights.shape) > 2:
#         attention_weights = attention_weights.mean(axis=tuple(range(len(attention_weights.shape)-2)))
    
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(attention_weights, cmap="viridis")
#     plt.title(title)
    
#     if save_path:
#         plt.savefig(save_path)
#         plt.close()
#         return save_path
    
#     return plt.gcf()

# # Then in the training loop:
# def log_attention_maps(attention_info, epoch, wandb_run=None):
#     """Log attention maps to wandb"""
#     if attention_info is None:
#         return
        
#     for key, value in attention_info.items():
#         if isinstance(value, dict):
#             for sub_key, attention_map in value.items():
#                 if isinstance(attention_map, torch.Tensor):
#                     fig_path = visualize_attention_weights(
#                         attention_map,
#                         f'{key}/{sub_key} Attention Weights',
#                         f'attention_{key}_{sub_key}_epoch_{epoch+1}.png'
#                     )
#                     if wandb_run:
#                         wandb_run.log({f"attention_maps/{key}/{sub_key}": wandb.Image(fig_path)})
        
#         elif isinstance(value, (list, tuple)):
#             for i, tensor in enumerate(value):
#                 if isinstance(tensor, torch.Tensor):
#                     fig_path = visualize_attention_weights(
#                         tensor,
#                         f'{key} Effect {i+1}',
#                         f'attention_{key}_effect_{i+1}_epoch_{epoch+1}.png'
#                     )
#                     if wandb_run:
#                         wandb_run.log({f"attention_maps/{key}/effect_{i+1}": wandb.Image(fig_path)})
        
#         elif isinstance(value, torch.Tensor):
#             fig_path = visualize_attention_weights(
#                 value,
#                 f'{key} Attention Weights',
#                 f'attention_{key}_epoch_{epoch+1}.png'
#             )
#             if wandb_run:
#                 wandb_run.log({f"attention_maps/{key}": wandb.Image(fig_path)})

# # Initialize wandb
# wandb.init(project='Multimodal_Alzheimer_nicara', config={
#     "learning_rate": 0.001,
#     "epochs": 50,
#     "batch_size": 2,
#     "alpha": 0.7
# })

