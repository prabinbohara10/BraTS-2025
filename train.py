from tqdm import tqdm
import torch
import torch.nn.functional as F
import wandb

from utils.utils import get_image_to_log

from monai.transforms import Compose, Activations
post_pred = Compose([Activations(softmax=True,dim=1)])

label_channel_names = ['necrotic', 'edema', 'enhancing']

# train function
def debug_train(data_loader, val_dataloader, model, optimizer, loss_fn, best_loss, patience, metrics, wandb_run_obj, max_epochs, device):
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        val_loss = 1.0
        val_metric = 0.0 

        # Reset all metrics
        metrics["train_dice_metric"].reset()
        metrics["precision_metric"].reset()
        metrics["recall_metric"].reset()
        metrics["nsd"].reset()
        
        batch_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
        
        for train_image_data, train_label_data in batch_bar:
            inputs, labels = train_image_data.to(device), train_label_data.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels) # this loss can have logits? check
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
    
            # model prediction postprocess:
            post_preded = post_pred(outputs)  # (B, C, H, W, D)
            confidences, argmax_preds = torch.max(post_preded, dim=1)  # (B, H, W, D)
            argmax_preds_flat = torch.reshape(argmax_preds, (argmax_preds.shape[0], -1)) # (B, H * W * D)
    
            preds_onehot = F.one_hot(argmax_preds, num_classes=4)  # (B, H, W, D, C)
            preds_onehot = preds_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, H, W, D)

            confidences_labels, argmax_labels = torch.max(labels, dim =1 ) # (B, H, W, D)
            argmax_labels_flat = torch.reshape(argmax_labels, (argmax_labels.shape[0], -1)) # (B, H * W * D)
            
            # Metics calculation:
            metrics["precision_metric"].update(argmax_preds_flat, argmax_labels_flat)
            metrics["recall_metric"].update(argmax_preds_flat, argmax_labels_flat)
            
            # Dice metric
            batch_dice = metrics["train_dice_metric"](y_pred=preds_onehot, y=labels)
            batch_nsd = metrics["nsd"](y_pred=preds_onehot, y=labels)
            # generate channelwise batch mean
            channel_1, channel_2, channel_3 = torch.mean(batch_dice,dim=0)
            nsd_c1, nsd_c2, nsd_c3 = torch.mean(batch_nsd, dim = 0)
            
            # Update batch tqdm bar
            batch_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Running Avg Loss": f"{train_loss / (batch_bar.n + 1):.4f}",
            })

            # batchwise stats
            wandb_run_obj.log({'train/batch/batch_loss': loss.item(),
                              f'train/batch/batch_dsc_{label_channel_names[0]}': channel_1.item(),
                              f'train/batch/batch_dsc_{label_channel_names[1]}': channel_2.item(),
                              f'train/batch/batch_dsc_{label_channel_names[2]}': channel_3.item(),
                              f'train/batch/batch_nsd_{label_channel_names[0]}': nsd_c1.item(),
                              f'train/batch/batch_nsd_{label_channel_names[1]}': nsd_c2.item(),
                              f'train/batch/batch_nsd_{label_channel_names[2]}': nsd_c3.item(),
                              f'train/batch/batch_precision': metrics["precision_metric"].compute().item(),
                              f'train/batch/batch_recall': metrics["recall_metric"].compute().item()
                              })
            
        train_loss /= len(data_loader)
        train_dsc = metrics["train_dice_metric"].aggregate().item()
        
        print(f"Epoch {epoch+1} - Epoch Loss: {train_loss:.4f} - Train Dice Score: {train_dsc:.4f}")

       
        # Validation per epoch
        model.eval()
        val_loss = 0

        # Reset all metrics
        metrics["train_dice_metric"].reset()
        metrics["precision_metric"].reset()
        metrics["recall_metric"].reset()
        metrics["nsd"].reset()
        
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, desc="Validating")
            #val_outputs, val_labels, metric = [], [], 0
            for val_image_data, val_label_data in val_bar:
                val_inputs, val_targets = val_image_data.to(device), val_label_data.to(device)
                val_outputs = model(val_inputs)
                
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                # model prediction postprocess:
                post_preded = post_pred(val_outputs)  # (B, C, H, W, D)
                confidences, argmax_preds = torch.max(post_preded, dim=1)  # (B, H, W, D)
                argmax_preds_flat = torch.reshape(argmax_preds, (argmax_preds.shape[0], -1)) # (B, H * W * D)
        
                preds_onehot = F.one_hot(argmax_preds, num_classes=4)  # (B, H, W, D, C)
                preds_onehot = preds_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, H, W, D)
    
                confidences_labels, argmax_labels = torch.max(val_targets, dim =1 ) # (B, H, W, D)
                argmax_labels_flat = torch.reshape(argmax_labels, (argmax_labels.shape[0], -1)) # (B, H * W * D)
                
                # Metics calculation:
                metrics["precision_metric"].update(argmax_preds_flat, argmax_labels_flat)
                metrics["recall_metric"].update(argmax_preds_flat, argmax_labels_flat)
                metrics["train_dice_metric"](y_pred=preds_onehot, y=val_targets)

            
                
            val_loss /= len(val_dataloader)

            val_dsc = metrics["train_dice_metric"].aggregate().item()
            val_precision = metrics["precision_metric"].compute().item()
            val_recall = metrics["recall_metric"].compute().item()

            print(f"Validation Loss: {val_loss:.4f} - Validation Dice Score: {val_dsc:.4f}")
            # lr_scheduler.step(val_dsc)
    
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
    
                best_model_path = f"best_model_epoch_{epoch}.pth"
                torch.save(model.state_dict(), best_model_path)
                
                artifact = wandb.Artifact(
                    name= wandb_run_obj.name,
                    type="model",
                    description=f"Checkpoint at epoch {epoch}"
                )
                artifact.add_file(best_model_path)
                wandb_run_obj.log_artifact(artifact)
                
                print("Best model saved both in local and as wanbd artifacts")
            else:
                patience_counter += 1
    
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        
        # logging images:
        # from training data
        inputs, labels = next(iter(data_loader))  # Take the first batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        image_number1 = 1
        input_img1, label_img1, pred_img1 = get_image_to_log(image_number1, inputs, labels, outputs, slice_idx = 64)

        # from validation data
        inputs, labels = next(iter(val_dataloader))  # Take the first batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
    
    
        image_number2 = 1
        input_img2, label_img2, pred_img2 = get_image_to_log(image_number2, inputs, labels, outputs, slice_idx = 64)
        #fig = get_image_to_log(image_number, inputs, labels, outputs, slice_idx=90)
    
        ##Logging once per epoch
        wandb_run_obj.log({
            "epoch" :epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_dsc": train_dsc,
            "val/val_loss": val_loss,
            "val/val_dsc": val_dsc,
            "val/val_precision": val_precision,
            "val/val_recall" : val_recall,
            "train_input_vs_gt_vs_pred": [
            wandb.Image(input_img1, caption="Train Input Slice"),
            wandb.Image(label_img1, caption="Train Ground Truth"),
            wandb.Image(pred_img1, caption="Train Prediction")
            ],
            "val_input_vs_gt_vs_pred": [
            wandb.Image(input_img2, caption="Val Input Slice"),
            wandb.Image(label_img2, caption="Val Ground Truth"),
            wandb.Image(pred_img2, caption="Val Prediction")
            ]
        }, commit=True)