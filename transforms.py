from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ToTensord,
    RandFlipd,
    RandRotate90d
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),  # Load NIfTI files
    EnsureChannelFirstd(keys=["label"]),   # Add channel dim to label (1, H, W, D)
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # Data augmentation
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),     # Random rotation
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),  # Normalize modalities
    ToTensord(keys=["image", "label"])    # Convert to PyTorch tensors
])


val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["label"]),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ToTensord(keys=["image", "label"])
])