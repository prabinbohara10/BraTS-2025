### Note:
- Current bt_processing does the cropping

```python
# Crop to desired shape
temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]
temp_mask = temp_mask[56:184, 56:184, 13:141]
```