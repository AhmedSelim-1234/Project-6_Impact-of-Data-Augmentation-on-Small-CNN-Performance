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
# none_transform = transforms.Compose(base_transform)
#
# # Combination 2: Flip only
# flip_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
# ] + base_transform)
#
# # Combination 3: Flip + Rotation
# flip_rotation_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
# ] + base_transform)
#
# # Combination 4: Flip + Rotation + Color
# flip_rotation_color_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.ColorJitter(
#        brightness=0.2,
#        contrast=0.2,
#        saturation=0.2,
#        hue=0.1
#     ),
# ] + base_transform)
#
# # Combination 5: Flip + Rotation + Color + Crop
# flip_rotation_color_crop_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.ColorJitter(
#         brightness=0.2,
#         contrast=0.2,
#         saturation=0.2,
#         hue=0.1
#     ),
#     transforms.RandomCrop(32, padding=4),
# ] + base_transform)
#
# # Combination 6: Full (everything including RandomErasing)
# # Note: RandomErasing MUST come after ToTensor
# full_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.ColorJitter(
#         brightness=0.2,
#         contrast=0.2,
#         saturation=0.2,
#         hue=0.1
#     ),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         (0.4914, 0.4822, 0.4465),
#         (0.2470, 0.2435, 0.2616)
#     ),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # after ToTensor!
# ])


none_transform = transforms.Compose(base_transform)
# ─────────────────────────────────────────
# One Augmentation Transformations
# ─────────────────────────────────────────

# Combination 2: Flip only
flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
] + base_transform)

# Combination 3: Rotation_deg_15
rotation_deg_15_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
] + base_transform)

# Combination 4: Rotation_deg_30
rotation_deg_30_transform = transforms.Compose([
    transforms.RandomRotation(degrees=30),
] + base_transform)

# Combination 5: Rotation_deg_90
rotation_deg_90_transform = transforms.Compose([
    transforms.RandomRotation(degrees=90),
] + base_transform)

# Combination 6: Color_0.2
color_pointTwo_transform = transforms.Compose([
    transforms.ColorJitter(
       brightness=0.2,
       contrast=0.2,
       saturation=0.2,
       hue=0.1
    ),
] + base_transform)

# Combination 7: Color_0.3
color_pointThree_transform = transforms.Compose([
    transforms.ColorJitter(
       brightness=0.3,
       contrast=0.3,
       saturation=0.3,
       hue=0.1
    ),
] + base_transform)

# Combination 8: Crop
crop_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
] + base_transform)

# ─────────────────────────────────────────
# Two Augmentation Transformations
# ─────────────────────────────────────────
#combination 1: flip_rotation deg 15
flip_rotation15_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
] + base_transform)

#combination 2: flip_crop
flip_crop_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
] + base_transform)

#combination 3: flip_color jitter 0.3
flip_color_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
       brightness=0.3,
       contrast=0.3,
       saturation=0.3,
       hue=0.1
    ),
] + base_transform)

#combination 4: rotation deg 15_ color
rotation_color_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
] + base_transform)

#combination 5: rotation deg 15_ crop
rotation_crop_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(32, padding=4),
] + base_transform)

#combination 6: color _ crop
color_crop_transform = transforms.Compose([
    transforms.ColorJitter(
       brightness=0.3,
       contrast=0.3,
       saturation=0.3,
       hue=0.1
    ),
    transforms.RandomCrop(32, padding=4),
] + base_transform)


# ─────────────────────────────────────────
# Three Augmentation Transformations
# ─────────────────────────────────────────
flip_color_crop_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
       brightness=0.3,
       contrast=0.3,
       saturation=0.3,
       hue=0.1
    ),
    transforms.RandomCrop(32, padding=4),
] + base_transform)

flip_rotation_crop_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(32, padding=4),
] + base_transform)

flip_rotation_color_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
       brightness=0.3,
       contrast=0.3,
       saturation=0.3,
       hue=0.1
    ),
] + base_transform)

rotation_color_crop_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
       brightness=0.3,
       contrast=0.3,
       saturation=0.3,
       hue=0.1
    ),
    transforms.RandomCrop(32, padding=4),
] + base_transform)

# ─────────────────────────────────────────
# Four Augmentation Transformations
# ─────────────────────────────────────────
flip_rotation_color_crop_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomCrop(32, padding=4),
] + base_transform)

# ─────────────────────────────────────────
# Smart Augmentation Transformations
# If a new dataset is used, then standard AutoAugment policies (ImageNet / CIFAR10 / SVHN)
# may or may not work well. That is one of the main limitations of AutoAugment.
# AutoAugment policies are dataset-specific.
# ─────────────────────────────────────────
auto_augment_transform = transforms.Compose([
    transforms.AutoAugment(
        policy = transforms.AutoAugmentPolicy.CIFAR10,
    ),
] + base_transform)

# ─────────────────────────────────────────
# DICTIONARY: select transform by name
# ─────────────────────────────────────────
TRANSFORMS = {
    # ______________________________________________________________________
    # one Transformations
    # ______________________________________________________________________
    # cutmix is being handled in train.py in combination_names
    'none': none_transform,
    'flip': flip_transform,
    'rotation_deg_15': rotation_deg_15_transform,
    'rotation_deg_30': rotation_deg_30_transform,
    'rotation_deg_90': rotation_deg_90_transform,
    'color_0.2':  color_pointTwo_transform,
    'color_0.3':  color_pointThree_transform,
    'crop': crop_transform,

    # ______________________________________________________________________
    # Two transformations
    # ______________________________________________________________________
    # cutmix is being handled in train.py in combination_names
    #'crop_cutmix':
    #'flip_cutmix':
    #'rotation15_cutmix':
    #'color0.3_cutmix':
    'flip_rotation': flip_rotation15_transform,
    'flip_crop': flip_crop_transform,
    'flip_color': flip_color_transform,
    'rotation_color': rotation_color_transform,
    'rotation_crop': rotation_crop_transform,
    'color_crop': color_crop_transform,

    # ______________________________________________________________________
    # Three transformations
    # ______________________________________________________________________
    # cutmix is being handled in train.py in combination_names
    #'rotation_crop_cutmix':,
    #'rotation_color_cutmix':,
    #'flip_color_cutmix':,
    #'flip_rotation_cutmix':,
    #'flip_crop_cutmix':,
    #'color_crop_cutmix':,
    'flip_color_crop': flip_color_crop_transform,
    'flip_rotation_crop': flip_rotation_crop_transform,
    'flip_rotation_color': flip_rotation_color_transform,
    'rotation_color_crop': rotation_color_crop_transform,

    # ______________________________________________________________________
    # Four & Five transformations
    # ______________________________________________________________________
    # 'flip_rotation_crop_cutmix'
    # 'flip_color_crop_cutmix'
    # 'rotation_color_crop_cutmix'
    # 'flip_rotation_color_cutmix'
    # 'flip_rotation_color_crop_cutmix'
    'flip_rotation_color_crop': flip_rotation_color_crop_transform,

    # ______________________________________________________________________
    # Smart transformations
    # ______________________________________________________________________
    'auto_augment': auto_augment_transform,
}

