# imports
import torch
import os
from loguru import logger
import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall

from monai.transforms import Compose, Activations
from monai.networks.nets import UNet, UNETR
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, SurfaceDiceMetric

# Custom imports
from dataset import PatchDataset
from train import debug_train


# Wandb setup
WANDB_API_KEY = "2ee0bb0e60f650d41059feeb5ab5ce243edf12c7"
wandb.login(key=WANDB_API_KEY)

# hyperparameters
batch_size = 2
img_size = (128,128,128) # if patch turned on, img_size=patch_size
patch_size = (64, 64, 64) # turned off the patching currently
num_workers = 4
num_label_classes = 4
img_channels = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"The selected device is : {device}")
max_epochs = 1000
best_model_path = "saved_model/best_model.pt"
label_channel_names = ['necrotic', 'edema', 'enhancing']
lr=1e-3

# early stopping:
best_loss = float('inf')
patience = 20



# -------------------- DATA LOADING --------------------
# Path to dataset (Modify as needed)
SPLIT_DIR = "data/split"

# Train and Validation Directories
train_img_dir = os.path.join(SPLIT_DIR, "train/images/")
train_mask_dir = os.path.join(SPLIT_DIR, "train/masks/")
val_img_dir = os.path.join(SPLIT_DIR, "val/images/")
val_mask_dir = os.path.join(SPLIT_DIR, "val/masks/")
test_img_dir = os.path.join(SPLIT_DIR, "test/images/")
test_mask_dir = os.path.join(SPLIT_DIR, "test/masks/")

# Step 2. Generate list of file paths for train , val, test and create dataloader
train_img_list, train_mask_list = os.listdir(train_img_dir), os.listdir(train_mask_dir)
val_img_list, val_mask_list = os.listdir(val_img_dir), os.listdir(val_mask_dir)
test_img_list, test_mask_list = os.listdir(test_img_dir), os.listdir(test_mask_dir)

train_dataset = PatchDataset(train_img_dir, train_img_list, train_mask_dir, train_mask_list, patch_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = PatchDataset(val_img_dir, val_img_list, val_mask_dir, val_mask_list, patch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dataset = PatchDataset(test_img_dir, test_img_list, test_mask_dir, test_mask_list, patch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# For debugging purpose: take 2 images, label pairs from the train set to test the pipeline
num_of_images_for_debug = 50
num_of_images_for_debug_val = int(num_of_images_for_debug * 0.2)
debug_train_img_dir = os.path.join(SPLIT_DIR, "train/images/")
debug_train_mask_dir = os.path.join(SPLIT_DIR, "train/masks/")

debug_train_img_list = sorted(os.listdir(debug_train_img_dir))[1:1+num_of_images_for_debug]
debug_train_mask_list = sorted(os.listdir(debug_train_mask_dir))[1:1+num_of_images_for_debug]
debug_val_img_list = sorted(os.listdir(val_img_dir))[1:1+num_of_images_for_debug_val]
debug_val_mask_list = sorted(os.listdir(val_mask_dir))[1:1+num_of_images_for_debug_val]

debug_train_dataset = PatchDataset(debug_train_img_dir, debug_train_img_list, debug_train_mask_dir, debug_train_mask_list, patch_size)
debug_train_dataloader = DataLoader(debug_train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

debug_val_dataset = PatchDataset(val_img_dir, debug_val_img_list, val_mask_dir, debug_val_mask_list, patch_size)
debug_val_dataloader = DataLoader(debug_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

logger.info(f"Len of Debug train dataset : {len(debug_train_dataset)} and Debug val dataset : {len(debug_val_dataset)}")


#define models, optimizers, losses, metrics
unet_model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,  
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
UNETR_model = UNETR(in_channels=4, out_channels=4, img_size=img_size, feature_size=16, hidden_size=768)

# Loss
combined_dice_ce_loss = DiceCELoss(include_background=True, softmax=True)
dice_loss = DiceLoss(softmax=True)
ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.25, 0.25, 0.25], device=device))


def total_loss(pred, target):
    return  dice_loss(pred, target) +  ce_loss(pred, target.squeeze(1))

# Optimizer & LR Scheduler
def get_optimizer(model_obj,lr=1e-3):
    return Adam(model_obj.parameters(), lr=lr)

# lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5)

# metrics
train_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
val_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
# TODO: Add Precision, Recall and other relevant ,...
# Instantiate epoch‐level metrics
precision_metric = MulticlassPrecision(num_classes=4, average='macro').to(device)  # :contentReference[oaicite:0]{index=0}
recall_metric    = MulticlassRecall(num_classes=4, average='macro').to(device)     # :contentReference[oaicite:1]{index=1}
nsd = SurfaceDiceMetric(class_thresholds = [10.0, 10.0, 10.0])

metrics = {
    "train_dice_metric" : train_dice_metric,
    "nsd" : nsd,
    "val_dice_metric" : val_dice_metric,
    "precision_metric" : precision_metric,
    "recall_metric" : recall_metric
}

post_pred = Compose([Activations(softmax=True,dim=1)])


# setting wandb
run = wandb.init(
    project="brats21-baseline-unetr",
    name="v5-unter-all_dataset",
    config={
        "epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
)
config = run.config

# models and training
model = UNETR_model.to(device)
#Check if multiple GPUs are available
# if torch.cuda.device_count() > 1:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     # Wrap the model with DataParallel
#     model = nn.DataParallel(model)
    
wandb.watch(model, log="all", log_freq=100)
optimizer = get_optimizer(model)
loss_fn = dice_loss
wandb_run_obj = run

debug_train(debug_train_dataloader, debug_val_dataloader, model, optimizer, loss_fn, best_loss, patience, metrics, wandb_run_obj, max_epochs, device)
