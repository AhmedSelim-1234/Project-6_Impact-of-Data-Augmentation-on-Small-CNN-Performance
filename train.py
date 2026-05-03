import numpy as np
from simple_model import SimpleCNN
from dataset import get_dataloader
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import v2

import copy
import json
import os

# _____________________ Device configuration _________________________
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ____________________________________________________
# Hyperparameters — all in one place for easy reference
# ____________________________________________________
NUM_EPOCHS    = 30
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 0.0005
NUM_CLASSES   = 10  # CIFAR-10 always has 10 classes

# cutmix (combine two images transformation)
cutmix = v2.CutMix(num_classes=10, alpha=1.0)
CUTMIX_PROB = 0.5

# ____________________________________________________
# Experiment settings
# ____________________________________________________
PERCENTAGES = [0.1, 0.25, 0.50, 1.0]

# _____________________________________________________
# List your combinations here
# make a new list variable for every combinations group (ones, twos, threes, fours, fives, smart)
# then assign this list to COMBINATIONS variable
# e.g: COMBINATIONS = ONES_COMBINATIONS
# _____________________________________________________
#ones combinations
ONES_COMBINATIONS = [
    'none',
    'none_cutmix',
    'flip',
    'rotation_deg_15',
    'rotation_deg_30',
    'rotation_deg_90',
    'color_0.2',
    'color_0.3',
    'crop'
]

#two combinations
TWO_COMBINATIONS = [
    'none',
    'crop_cutmix',
    'flip_rotation',
    'flip_crop',
    'flip_cutmix',
    'flip_color',
    'rotation_color',
    'rotation_crop',
    'rotation_deg_15_cutmix',
    'color_0.3_cutmix',
    'color_crop',
]

#three combinations
#write here:
THREE_COMBINATIONS = [
    'none',
    'rotation_crop_cutmix',
    'rotation_color_cutmix',
    'rotation_color_crop',
    'flip_color_cutmix',
    'flip_rotation_cutmix',
    'flip_crop_cutmix',
    'flip_color_crop',
    'flip_rotation_crop',
    'flip_rotation_color',
    'color_crop_cutmix',
]
#four & five combinations
#write here:
FOUR_AND_FIVE_COMBINATIONS = [
    'none',
    'flip_rotation_crop_cutmix',
    'flip_color_crop_cutmix',
    'rotation_color_crop_cutmix',
    'flip_rotation_color_cutmix',
    'flip_rotation_color_crop',
    'flip_rotation_color_crop_cutmix',
]

SMART_COMBINATIONS = [
    'none',
    'auto_augment',
]
# COMBINATIONS = ONES_COMBINATIONS
# COMBINATIONS = TWO_COMBINATIONS
# COMBINATIONS = THREE_COMBINATIONS
# COMBINATIONS = FOUR_AND_FIVE_COMBINATIONS
COMBINATIONS = SMART_COMBINATIONS


# ____________________________________________________
# Training & Validation Functions
# ____________________________________________________
def train_epoch(model, train_loader, loss_function, optimizer, device, use_cutmix=False):
    """
    Performs a single training epoch.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
        loss_function (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU) to perform training on.

    Returns:
        float: The average training loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    running_loss = 0.0
    # Iterate over batches of data in the training loader
    for images, labels in train_loader:
        # Move images and labels to the specified device
        images, labels = images.to(device), labels.to(device)

        #apply cutmix (mixing two images) if required:
        if use_cutmix and np.random.rand() < 0.5:
            images, labels = cutmix(images, labels)

        # Clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Perform a forward pass to get model outputs
        outputs = model(images)

        # Calculate the loss
        loss = loss_function(outputs, labels)

        # Perform a backward pass to compute gradients
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the training loss for the batch
        running_loss += loss.item() * images.size(0)

    # Calculate and return the average training loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def validate_epoch(model, val_loader, loss_function, device):
    """
    Performs a single validation epoch.

    Args:
        model (torch.nn.Module): The neural network model to validate.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        loss_function (callable): The loss function.
        device (torch.device): The device (CPU or GPU) to perform validation on.

    Returns:
        tuple: A tuple containing the average validation loss and validation accuracy.
    """
    # Set the model to evaluation mode
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    ### START CODE HERE ###

    # Disable gradient calculations for validation
    with torch.no_grad():

    ### END CODE HERE ###

        # Iterate over batches of data in the validation loader
        for images, labels in val_loader:
            # Move images and labels to the specified device
            images, labels = images.to(device), labels.to(device)

            ### START CODE HERE ###

            # Perform a forward pass to get model outputs
            outputs = model(images)

            # Calculate the validation loss for the batch
            val_loss = loss_function(outputs, labels)

            # Accumulate the validation loss
            running_val_loss += val_loss.item()

            # Get the predicted class labels
            _, predicted = torch.max(outputs, dim=1)

            ### END CODE HERE ###

            # Update the total number of samples
            total += labels.size(0)

            # Update the number of correct predictions
            correct += (predicted == labels).sum().item()


    # Calculate the average validation loss and accuracy for the epoch
    epoch_val_loss = running_val_loss / len(val_loader.dataset)
    epoch_accuracy = 100.0 * correct / total

    return epoch_val_loss, epoch_accuracy

def training_loop(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device, use_cutmix = False):
    """
    Trains and validates a PyTorch neural network model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        loss_function (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimization algorithm.
        num_epochs (int): The total number of epochs to train for.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run training on.

    Returns:
        tuple: A tuple containing the best trained model and a list of metrics
               (train_losses, val_losses, val_accuracies).
    """
    # Move the model to the specified device (CPU or GPU)
    model.to(device)

    # Initialize variables to track the best performing model
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0

    # Initialize lists to store training and validation metrics
    train_losses, val_losses, val_accuracies = [], [], []

    print("--- Training Started ---")

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Perform one epoch of training
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device, use_cutmix)
        train_losses.append(epoch_loss)

        # Perform one epoch of validation
        epoch_val_loss, epoch_accuracy = validate_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)

        # Print the metrics for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")

        # Check if the current model is the best one so far
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            # Save the state of the best model in memory
            best_model_state = copy.deepcopy(model.state_dict())

    print("--- Finished Training ---")

    # Load the best model weights before returning
    if best_model_state:
        print(f"\n--- Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)

    # Consolidate all metrics into a single list
    metrics = [train_losses, val_losses, val_accuracies]

    # Return the trained model and the collected metrics
    return model, metrics




def run_experiment(percentage, augmentation_combo_name):
    """
    Runs one complete experiment.

    Args:
        percentage:               0.10, 0.25, 0.50, or 1.0
        augmentation combo_name: 'none','flip','flip_rotation',
                                 'flip_rotation_color',
                                 'flip_rotation_color_crop','full'

    Returns:
        final accuracy (float) , metrics (list)
    """
    print(f"\n{'='*50}")
    print(f"Experiment: {percentage*100:.0f}% data, {augmentation_combo_name}")
    print(f"{'='*50}")

    #check whether cutmix exists or not
    use_cutmix = 'cutmix' in augmentation_combo_name
    base_combo_name = augmentation_combo_name.replace('_cutmix', '')

    # Get data
    train_loader, test_loader = get_dataloader(
        percentage,
        batch_size=BATCH_SIZE,
        augmentation_combo_name= base_combo_name
        # augmentation_combo_name=augmentation_combo_name,
    )

    # model for each experiment
    model = SimpleCNN(num_classes=NUM_CLASSES)

    # Loss and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Train
    trained_model, metrics = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=device,
        use_cutmix=use_cutmix
    )

    # Final accuracy = best accuracy achieved
    best_accuracy = max(metrics[2])
    print(f"\nFinal Best Accuracy: {best_accuracy:.2f}%")

    return best_accuracy, metrics



# _________________________________________
# RUN ALL EXPERIMENTS
# 4 percentages × N combinations = 4N
# _________________________________________
from plot_helper_functions import (plot_experiment,
                                   plot_bar_per_percentage,
                                   plot_final_summary,
                                   plot_all_combos_per_percentage
                                   )

# After running all experiments store metrics:
all_metrics = {} #train loss + validate loss + validate accuracy per epoch
all_results = {} #stores only best accuracy per experiment

os.makedirs('results', exist_ok=True)

for percentage in PERCENTAGES:
    for combination_name in COMBINATIONS:

        # Run experiment
        accuracy, metrics = run_experiment(percentage, combination_name)

        # Store results
        key = f"{int(percentage*100)}%_{combination_name}"
        all_results[key] = accuracy
        all_metrics[key] = metrics

        # Save to disk immediately
        with open(f'results/{key}.json', 'w') as f:
            json.dump({
                'train_losses': metrics[0],
                'val_losses': metrics[1],
                'val_accuracies': metrics[2],
                'best_accuracy': accuracy,
                'percentage': percentage,
                'combo': combination_name
            }, f, indent=2)
        print(f"Saved: results/{key}.json")

        # Plot this experiment's learning curves
        plot_experiment(metrics, percentage, combination_name)

# _________________________________________
# PLOTS
# _________________________________________

# Plot 1: All 6 combos on same graph for each percentage (4 graphs)
plot_all_combos_per_percentage(
    all_metrics, all_results, PERCENTAGES, COMBINATIONS)

# Plot 2: Bar chart per percentage (4 bar charts)
plot_bar_per_percentage(all_results, PERCENTAGES, COMBINATIONS)

# Plot 3: THE main summary graph (1 graph, goes in report!)
# Plot the big final summary
#for all percentages (it will produce error if we didn't pass the 4 percentages[0.1, 0.25, 0.5, 1.0] experiments
#so comment this line if you will try one percentage only (e.g: 0.1)
plot_final_summary(all_results, PERCENTAGES, COMBINATIONS)


# _________________________________________
# PRINT FULL RESULTS TABLE
# _________________________________________
print("\n" + "=" * 55)
print("FULL RESULTS SUMMARY")
print("=" * 55)
print(f"{'Experiment':<35} {'Best Accuracy':>15}")
print("-" * 55)

for percentage in PERCENTAGES:
    p = int(percentage * 100)
    print(f"\n--- {p}% Training Data ---")
    for combo in COMBINATIONS:
        key = f"{p}%_{combo}"
        acc = all_results[key]
        print(f"  {combo:<33} {acc:>8.2f}%")

print("=" * 55)

# Save final summary table
with open('results/final_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("\nSaved: results/final_summary.json")
