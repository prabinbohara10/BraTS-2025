# test models, optimizers, losses

# test fixtures


# generate dummy image data, pass through model and test if the expected shape is correct and no error is thrown

# generate dummy model prediction and labels and pass through loss functions and see if any error is thrown
label_size = (batch_size,num_label_classes,*img_size)
model_pred_logits = torch.randn(label_size)
sample_labels = torch.zeros_like(model_pred_logits,dtype=torch.long)
model_pred_probs = post_pred(model_pred_logits)
assert torch.all(model_pred_probs >= 0.) 
assert torch.all(model_pred_probs <= 1.)

# test if we get an error
loss_val = dice_loss(model_pred_logits,sample_labels)


# test for imperfect prediction
batch_imgs, batch_labels = next(iter(debug_train_dataloader))
batch_preds = unet_model(batch_imgs)
loss_val = dice_loss(batch_preds, batch_labels)
assert loss_val.item() >= 0.
assert loss_val.item() <= 1.
batch_probs = post_pred(batch_preds)
assert torch.abs(torch.sum(batch_probs[0,:,0,0,0]) - 1.0) < 0.01


# test dice metric func
# postprocess
confidences, argmax_preds = torch.max(batch_probs, dim=1)  # (B, H, W, D)
    
preds_onehot = F.one_hot(argmax_preds, num_classes=4)  # (B, H, W, D, C)
preds_onehot = preds_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, H, W, D)
train_dice_metric = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)
batch_dice_metric = train_dice_metric(preds_onehot, batch_labels )
print(batch_dice_metric)
aggregate_dice_metric = train_dice_metric.aggregate()
assert len(aggregate_dice_metric.shape) == 1, print(f'expected scalar but got {batch_dice_metric}')
assert aggregate_dice_metric > 0.0, print(f'expected scalar but got {batch_dice_metric}')
assert aggregate_dice_metric < 1.0, print(f'expected scalar but got {batch_dice_metric}')

# test for perfect prediction
#perfect_logits = torch.zeros_like(model_pred_logits,dtype=torch.float32)
#assert torch.all(post_pred(perfect_logits) < 0.01), print(f'')
#loss_val = dice_loss(perfect_logits, sample_labels)
#assert torch.abs(loss_val) < 0.001, print(f'expected 0.0 got {loss_val.item():.2f}')

# test for precision and recall
argmax_preds = torch.zeros([8,6,6,6]).to(device)
argmax_labels = torch.zeros([8,6,6,6]).to(device)
metrics["train_dice_metric"].reset()
metrics["precision_metric"].reset()
precision = metrics["precision_metric"].update(argmax_preds, argmax_labels)
print("pre :", precision)
assert metrics["precision_metric"].compute() == 1.0

#test nsd:
batch_dice_metric = metrics["nsd"](preds_onehot, batch_labels)
metrics["nsd"].aggregate().item()


precision_metric = MulticlassPrecision(num_classes=4, average=None).to(device)
recall_metric    = MulticlassRecall(num_classes=4, average=None).to(device)  
