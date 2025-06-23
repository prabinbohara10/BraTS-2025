import glob
import os
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#
TRAIN_DATASET_DIR = "data/original/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData/Validation"
PROCESSED_DIR = "data/processsed_validation"

# Initialize scaler
scaler = MinMaxScaler()

# Create output folders
os.makedirs(os.path.join(PROCESSED_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "masks"), exist_ok=True)

# List all patient folders
patient_folders = glob.glob(os.path.join(TRAIN_DATASET_DIR, '*'))

print(f"Total MRI cases : {len(patient_folders)}")

total_success = 0
total_fail = 0

# Loop over each patient folder
for img, folder in enumerate(patient_folders):
    # Construct paths
    t1n_path = glob.glob(os.path.join(folder, '*t1n.nii.gz'))
    t1c_path = glob.glob(os.path.join(folder, '*t1c.nii.gz'))
    t2f_path = glob.glob(os.path.join(folder, '*t2f.nii.gz'))
    t2w_path = glob.glob(os.path.join(folder, '*t2w.nii.gz'))
    mask_path = glob.glob(os.path.join(folder, '*seg.nii.gz'))

    filename = os.path.basename(os.path.dirname(t1c_path[0]))

    # Check if all required modalities are present
    if not (t1n_path and t1c_path and t2f_path and t2w_path):
        print(f"[SKIP] Missing modalities in folder: {folder}")
        continue

    print("Now preparing image and masks number:", img)

    # Load and scale modalities
    def load_and_scale(img_path):
        img_data = nib.load(img_path[0]).get_fdata()
        img_data = scaler.fit_transform(img_data.reshape(-1, img_data.shape[-1])).reshape(img_data.shape)
        return img_data

    try:
        temp_image_t1n = load_and_scale(t1n_path)
        temp_image_t1c = load_and_scale(t1c_path)
        temp_image_t2f = load_and_scale(t2f_path)
        temp_image_t2w = load_and_scale(t2w_path)

        # Combine modalities
        temp_combined_images = np.stack([temp_image_t1n, temp_image_t1c, temp_image_t2f, temp_image_t2w], axis=3)

        # Crop to desired shape
        temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]

        # Save images
        np.save(f'{PROCESSED_DIR}/images/image_{filename}.npy', temp_combined_images)

        # Process mask only if it exists
        if mask_path:
            temp_mask = nib.load(mask_path[0]).get_fdata()
            temp_mask = temp_mask.astype(np.uint8)
            temp_mask[temp_mask == 4] = 3  # Reassign mask value 4 to 3
            temp_mask = temp_mask[56:184, 56:184, 13:141]

            val, counts = np.unique(temp_mask, return_counts=True)

            if len(counts) > 1 and (1 - (counts[0] / counts.sum())) > 0.01:
                temp_mask = to_categorical(temp_mask, num_classes=4)
                np.save(f'{PROCESSED_DIR}/masks/mask_{filename}.npy', temp_mask)
                total_success += 1
            else:
                print("I am not a good addition to the model")
                total_fail += 1
        else:
            print("[INFO] No mask found for validation data.")
            total_success += 1  # Consider it a success for validation data without masks

    except Exception as e:
        print(f"[ERROR] Failed to process {folder}: {e}")
        continue

print(f"\nTotal MRI cases : {len(patient_folders)}")
print(f"Total Success cases : {total_success}")
print(f"Total failed cases : {total_fail}")