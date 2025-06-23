import os
import shutil
import random

def transfer(file_list, split_name, image_dir, mask_dir, output_base):
    """Transfer files to the specified split directory."""
    for image_file in file_list:
        image_index = image_file.replace('image_', '')
        mask_file = f"mask_{image_index}"

        src_image = os.path.join(image_dir, image_file)
        src_mask = os.path.join(mask_dir, mask_file)
        dst_image = os.path.join(output_base, split_name, 'images', image_file)
        dst_mask = os.path.join(output_base, split_name, 'masks', mask_file)

        if os.path.exists(src_image) and os.path.exists(src_mask):
            shutil.copy(src_image, dst_image)
            print(f"Copying: {src_image}  to {dst_image}")
            shutil.copy(src_mask, dst_mask)
            print(f"Copying: {src_mask}  to {dst_mask}")
        else:
            print(f"Skipping: {image_file} or corresponding mask not found.")

def setup_directories(output_base, splits):
    """Create directories for dataset splits."""
    for split in splits:
        os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, 'masks'), exist_ok=True)

def split_dataset(image_files, train_ratio, val_ratio, test_ratio):
    """Split dataset into train, validation, and test sets."""
    total = len(image_files)
    if test_ratio > 0:
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]

        return {'train': train_files, 'val': val_files, 'test': test_files}
    else:
        split_point = int(train_ratio * total)
        train_files = image_files[:split_point]
        val_files = image_files[split_point:]

        return {'train': train_files, 'val': val_files}

if __name__ == "__main__":
    base_dir = "data/processed"
    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")
    output_base = "data/split"
    
    train_ratio = 0.7
    val_ratio = 0.3
    # If you want to include a test set, set test_ratio > 0
    test_ratio = 0.0

    image_files = sorted(os.listdir(image_dir))
    random.seed(42)
    random.shuffle(image_files)

    active_splits = split_dataset(image_files, train_ratio, val_ratio, test_ratio)
    setup_directories(output_base, active_splits)

    for split_name, file_list in active_splits.items():
        transfer(file_list, split_name, image_dir, mask_dir, output_base)

    print("Data split complete:")
    for split, files in active_splits.items():
        print(f"{split.capitalize()}: {len(files)} samples")
