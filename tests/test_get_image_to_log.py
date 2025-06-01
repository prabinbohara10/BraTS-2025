# test get_image_to_log

batch_images, batch_labels = next(iter(train_dataloader))
print(f"Train image shape: {sample_image.shape}")  
print(f"Train label shape: {sample_label.shape}")


vis_img, vis_label, vis_pred = get_image_to_log(1, batch_imgs, batch_labels, batch_preds,slice_idx=64)
visualize_img_label_pred_slice(vis_img, vis_label, vis_pred)

# test get_image_to_log

batch_images, batch_labels = next(iter(val_dataloader))
print(f"Train image shape: {sample_image.shape}")  
print(f"Train label shape: {sample_label.shape}")


vis_img, vis_label, vis_pred = get_image_to_log(1, batch_imgs, batch_labels, batch_preds,slice_idx=64)
visualize_img_label_pred_slice(vis_img, vis_label, vis_pred)