import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset
from augmentations import *

def get_datasets(augmentation_combo_name='none'):
    """
    Downloads CIFAR-10 and returns full train and test datasets.
    First time: downloads automatically to './data' folder
    After that: loads from disk (no re-download)
    """
    transform = TRANSFORMS[augmentation_combo_name]

    full_train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    full_test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=none_transform #no augmentation in test
    )

    return full_train_dataset, full_test_dataset


def get_subset(dataset, percentage, seed = 42):
    """
     Takes only a percentage of a dataset.

      Args:
        dataset:    the full dataset
        percentage: 0.10 for 10%, 0.25 for 25%, etc.
        seed:       random seed (keeps results reproducible)

      Returns:
        A subset of the dataset
    """
    # How many images to keep
    total_size = len(dataset) #50,000
    subset_size = int(total_size * percentage) # 5,000 for 10%

    # Pick random indices (which images to keep)
    # seed=42 means we always pick the SAME images
    # so experiments are reproducible
    np.random.seed(seed)
    indices = np.random.choice(total_size, subset_size, replace=False)

    #create subset using these indices
    subset = Subset(dataset, indices)

    return subset

def get_dataloader(percentage, batch_size= 32, augmentation_combo_name='none'):
    """
    Main function — call this from your training code.

    Usage:
        train_loader, test_loader = get_dataloaders(0.10)  # 10%
        train_loader, test_loader = get_dataloaders(0.25)  # 25%
        train_loader, test_loader = get_dataloaders(0.50)  # 50%
        train_loader, test_loader = get_dataloaders(1.0)   # 100%

        # No augmentation:
        train_loader, test_loader = get_dataloader(0.10, combo_name='none')

        # Full augmentation:
        train_loader, test_loader = get_dataloader(0.10, combo_name='full')

    Args:
        percentage: 0.10, 0.25, 0.50, or 1.0
        batch_size: how many images per batch (32 is standard)
        augmentation_combo_name: one of the 6 combinations:
                'none', 'flip', 'flip_rotation',
                'flip_rotation_color',
                'flip_rotation_color_crop', 'full'

    Returns:
        train_loader, test_loader

    """

    #Get full datasets
    full_train_dataset, test_dataset = get_datasets(augmentation_combo_name)

    # Take only the percentage we want for training
    train_subset = get_subset(full_train_dataset, percentage)

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True          # shuffle training data every epoch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False         # no need to shuffle test data
    )

    # Print info so we know what's happening
    print(f"Training on {len(train_subset)} images ({percentage*100:.0f}% of training data)")
    print(f"Testing on  {len(test_dataset)} images (always full test set)")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, test_loader