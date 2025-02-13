import kagglehub

# Download latest version
path = kagglehub.dataset_download("debeshjha1/polypgen-video-sequence")

print("Path to dataset files:", path)

import shutil
import os

# Define destination (current directory)
destination = os.getcwd()

# Copy dataset to current directory
shutil.copytree(path, os.path.join(destination,"datasets", "polypgen"), dirs_exist_ok=True)

print(f"Dataset copied to: {destination}/datasets/polypgen")


# Define the root directory (where polypgen is located)
root_dir = f"{destination}/datasets/polypgen"

# Define Training and Test directories
train_images_dir = os.path.join(root_dir, "Training/images")
train_masks_dir = os.path.join(root_dir, "Training/masks")
test_images_dir = os.path.join(root_dir, "Test/images")
test_masks_dir = os.path.join(root_dir, "Test/masks")

# Create necessary directories
for folder in [train_images_dir, train_masks_dir, test_images_dir, test_masks_dir]:
    os.makedirs(folder, exist_ok=True)

# Iterate through all seq<i> folders
for i in range(1, 24):
    seq_folder = os.path.join(root_dir,"positive_cropped", f"seq{i}")

    if os.path.exists(seq_folder):
        images_src = os.path.join(seq_folder, "images")
        masks_src = os.path.join(seq_folder, "masks")

        if 1 <= i <= 18:  # Training Data
            images_dst = os.path.join(train_images_dir, str(i))
            masks_dst = os.path.join(train_masks_dir, str(i))
        else:  # Test Data (19-23)
            images_dst = os.path.join(test_images_dir, str(i))
            masks_dst = os.path.join(test_masks_dir, str(i))

        # Copy images and masks directories
        if os.path.exists(images_src):
            shutil.copytree(images_src, images_dst, dirs_exist_ok=True)
        if os.path.exists(masks_src):
            shutil.copytree(masks_src, masks_dst, dirs_exist_ok=True)

        # Remove the original seq<i> folder
        shutil.rmtree(seq_folder)
        print(f"Processed and deleted: {seq_folder}")

print("Dataset organization complete!")
