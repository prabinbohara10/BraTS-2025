import os
import torch
from torchvision import transforms
from PIL import Image

from dataset import PatchDataset
from torch.utils.data import DataLoader
from monai.networks.nets import UNETR
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torchio as tio
import nibabel as nib

img_size = (128,128,128)
patch_size = (64, 64, 64)
batch_size = 2
num_workers = 4
output_dir = "outputs"

# -------------------- DATA LOADING --------------------
# Path to dataset (Modify as needed)
SPLIT_DIR = "data/processsed_validation"

ORIGINAL_IMAGE_PATH = "data/original/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData/Validation/BraTS-SSA-00125-000/BraTS-SSA-00125-000-t1c.nii.gz"
# Train and Validation Directories
val_img_dir = os.path.join(SPLIT_DIR, "images/")
val_img_list = os.listdir(val_img_dir)

val_dataset = PatchDataset(val_img_dir, val_img_list, patch_size=patch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(in_channels=4, out_channels=4, img_size=img_size, feature_size=16, hidden_size=768)
# Load the trained model
model.load_state_dict(torch.load("saved_models/best_model_epoch_78.pth", map_location=device))
model.eval()
model.to(device)

original_image = nib.load(ORIGINAL_IMAGE_PATH)

with torch.no_grad():
    for image, filename in val_dataloader:
        output = model(image.to(device))
        predicted = torch.argmax(output, dim=1).cpu().numpy()
        
        for pred, fname in zip(predicted, filename):
            pred_np = pred.astype(np.uint8)
            
            # Expand to 3D volume: [1, H, W] for NIfTI
            # pred_3d = np.expand_dims(pred_np, axis=0)

            # resize = tio.transforms.Resize((240, 240, 155))
            # resized = resize(tio_image)
            # resized_np = resized.tensor.squeeze().numpy()

            tio_image = tio.ScalarImage(tensor=pred_np[np.newaxis, ...])
            transform = tio.CropOrPad((240,240,155))
            output = transform(tio_image)
            pred_3d = output.tensor.squeeze().numpy()

            # Save as NIfTI
            nii_img = nib.Nifti1Image(pred_3d, original_image.affine)
            base_name = os.path.splitext(fname)[0]
            out_path = os.path.join(output_dir, base_name + ".nii.gz")
            os.makedirs(output_dir, exist_ok=True)
            nib.save(nii_img, out_path)
            print("Saved:", base_name)
