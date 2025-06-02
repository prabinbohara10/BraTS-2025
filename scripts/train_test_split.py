import splitfolders
import os

SPLIT_DIR = "data/split"
PROCESSED_DIR = "data/processsed"

os.makedirs(SPLIT_DIR, exist_ok=True)

input_folder = PROCESSED_DIR
output_folder = SPLIT_DIR
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .15, .10), group_prefix=None)