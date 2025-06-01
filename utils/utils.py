# imports
import os
import numpy as np
from scipy.ndimage import rotate
import torch
import matplotlib.pyplot as plt

# function to extract a random patch of size 'patch_size' from the original image 
def extract_patch(image, mask, patch_size):
    img_shape = image.shape[:3]
    patch_x = np.random.randint(0, max(img_shape[0] - patch_size[0], 1))
    patch_y = np.random.randint(0, max(img_shape[1] - patch_size[1], 1))
    patch_z = np.random.randint(0, max(img_shape[2] - patch_size[2], 1))
   
    return (
        image[patch_x:patch_x + patch_size[0], patch_y:patch_y + patch_size[1], patch_z:patch_z + patch_size[2], :],
        mask[patch_x:patch_x + patch_size[0], patch_y:patch_y + patch_size[1], patch_z:patch_z + patch_size[2]])


# functions for augmentations
def gamma_correction(image, gamma):
    return np.clip(image ** gamma, 0, 1)

def augment_image(image, mask, is_training=True):
    if is_training:
        # Rotation
        angle = np.random.uniform(-15, 15)
        image = rotate(image, angle, axes=(0, 1), reshape=False, mode='reflect')
        mask = rotate(mask, angle, axes=(0, 1), reshape=False, mode='reflect')
        
        # Flipping
        if np.random.rand() > 0.5:
            image, mask = np.flip(image, axis=0).copy(), np.flip(mask, axis=0).copy()
        if np.random.rand() > 0.5:
            image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
     
        # Brightness Adjustment
        brightness = np.random.uniform(0.9, 1.1)
        image = np.clip(image * brightness, 0, 1)

        # Noise Addition (Gaussian noise)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)

# Validate directories exist
def validate_dir(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")

# vis utils
def get_image_to_log(image_number, inputs, labels, outputs, slice_idx=100, verbose=False):
    """
    Visualize a single batch with input, ground truth (label), and prediction images for a selected slice.

    Args:
    - image_number: index of the batch
    - inputs: The input tensor (B, C, H, W, D)
    - labels: The ground truth tensor (B, C, H, W, D)
    - outputs: The model outputs (B, C, H, W, D)
    - slice_idx: The index of the slice to visualize (default 100)

    Returns:
    - None
    """
    # Get the first sample in the batch
    input_img = inputs[image_number, 0, :, :, slice_idx].cpu().numpy()  # Use the first image in the batch
    #label_img = labels[0, 0, :, :, slice_idx].cpu().numpy()  # Use the first label in the batch

    # Apply softmax to labels
    # BCHWD -> BHWD
    label_img = torch.argmax(labels, dim=1)[image_number, :, :, slice_idx].cpu().numpy()  # First prediction in the batch
    
    # Apply softmax to outputs and get the prediction
    softmaxed = torch.nn.functional.softmax(outputs, dim=1)
    pred_img = torch.argmax(softmaxed, dim=1)[image_number, :, :, slice_idx].cpu().numpy()  # First prediction in the batch

    if verbose:
        print(input_img.shape, label_img.shape, pred_img.shape)

    # Define a colormap (class 0 to 3)
    palette = {
        0: (0, 0, 0),         # background - black
        1: (255, 0, 0),       # class 1 - red
        2: (0, 255, 0),       # class 2 - green
        3: (0, 0, 255),       # class 3 - blue
    }
    
    # Convert to RGB
    def label_to_rgb(label, palette):
        h, w = label.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in palette.items():
            rgb[label == cls] = color
        return rgb
    
    label_img = label_to_rgb(label_img, palette)
    pred_img = label_to_rgb(pred_img, palette)
    
    return input_img, label_img, pred_img
    
def visualize_img_label_pred_slice(input_img, label_img, pred_img):
    # Visualize the input, label, and prediction side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input Image in grayscale
    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title("Input Image")

    # Ground Truth using custom colormap
    axes[1].imshow(label_img,  interpolation='none')
    axes[1].set_title("Ground Truth")

    # Prediction using custom colormap
    axes[2].imshow(pred_img, interpolation='none')
    axes[2].set_title("Prediction")

    plt.show()

# function to load numpy file given a filepath
def load_npy(filepath):
    try:
        return np.load(filepath, allow_pickle=True).astype(np.float32)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None