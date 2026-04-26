import torch
import torchvision
import torchvision.transforms as transforms




# ─────────────────────────────────────────
# BASE: always needed (convert + normalize)
# ─────────────────────────────────────────
base_transform = [
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    ),
]

# ─────────────────────────────────────────
# 6 AUGMENTATION COMBINATIONS
# Each one adds more augmentation than the last
# ─────────────────────────────────────────

# Combination 1: No augmentation (baseline)
none_transform = transforms.Compose(base_transform)

# Combination 2: Flip only
flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
] + base_transform)

# Combination 3: Flip + Rotation
flip_rotation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
] + base_transform)

# Combination 4: Flip + Rotation + Color
flip_rotation_color_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
       brightness=0.2,
       contrast=0.2,
       saturation=0.2,
       hue=0.1
    ),
] + base_transform)

# Combination 5: Flip + Rotation + Color + Crop
flip_rotation_color_crop_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomCrop(32, padding=4),
] + base_transform)

# Combination 6: Full (everything including RandomErasing)
# Note: RandomErasing MUST come after ToTensor
full_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    ),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # after ToTensor!
])

# ─────────────────────────────────────────
# DICTIONARY: select transform by name
# ─────────────────────────────────────────
TRANSFORMS = {
    'none': none_transform,
    'flip': flip_transform,
    'flip_rotation': flip_rotation_transform,
    'flip_rotation_color': flip_rotation_color_transform,
    'flip_rotation_color_crop': flip_rotation_color_crop_transform,
    'full': full_transform,
}
