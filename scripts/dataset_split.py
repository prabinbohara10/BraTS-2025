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
            shutil.copy(src_mask, dst_mask)
        else:
            print(f"Skipping: {image_file} or corresponding mask not found.")

def create_split_directories(output_base, splits):
    """Create directories for dataset splits."""
    for split in splits:
        os.makedirs(os.path.join(output_base, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base, split, 'masks'), exist_ok=True)

def split_dataset(image_dir, mask_dir, output_base, train_ratio, val_ratio, test_ratio):
    """Split dataset into train, validation, and test sets."""
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"The directory '{image_dir}' does not exist. Please check the path.")

    image_files = sorted(os.listdir(image_dir))
    random.seed(42)
    random.shuffle(image_files)

    total = len(image_files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    transfer(train_files, 'train', image_dir, mask_dir, output_base)
    transfer(val_files, 'val', image_dir, mask_dir, output_base)
    transfer(test_files, 'test', image_dir, mask_dir, output_base)

    print(f"Dataset split completed: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

if __name__ == "__main__":
    base_dir = "data/processed"
    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")
    output_base = "data/split"
    splits = ['train', 'val', 'test']

    create_split_directories(output_base, splits)

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    split_dataset(image_dir, mask_dir, output_base, train_ratio, val_ratio, test_ratio)
