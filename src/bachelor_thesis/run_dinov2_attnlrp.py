from mask_generator import MaskGenerator
from utils import load_all_configs
from dataset import GorillaReIDDataset
# 1) run sam to get segmentation masks of data split, if not already saved
# 2) create dataset with masks
# 3) run lrp with swept parameters on images in dataset and compute the mask score
# 3a) compare basemodel vs finetuned model on val
# 3b) compare finetuned model on train vs val (overfitting?)
# 4) save worst performing images and mask their background. How does the knn score change? can I also recompute accuracy with only these few images?
if __name__ == "__main__":
    MASK_TRANSFORM = transforms.Compose([
        transforms.Resize(
            size=data_config['input_size'][1:], # e.g., (518, 518)
            interpolation=transforms.InterpolationMode.NEAREST # Use NEAREST for masks!
        ),
        transforms.ToTensor(), # Converts mask to a [1, H, W] tensor of floats (0.0 or 1.0)
    ])
    # IMPORTANT: The interpolation mode for the mask must be NEAREST.
    # Using BILINEAR or BICUBIC would create intermediate values (like 0.5)
    # along the edges, blurring the mask.
    mask_generator = MaskGenerator(
        model_checkpoint_path=cfg["model"]["sam2_checkpoint_path"],
        model_config_dir=cfg["model"]["sam2_config_dir"],
    )
    val_dataset = GorillaReIDDataset(
        image_dir=val_dir,
        filenames=val_filenames,
        transform=image_transform,       # The full transform for the image
        mask_dir=val_mask_dir,
        mask_transform=mask_transform,  # The spatial-only transform for the mask
        generate_masks_from="cropped",  
        mask_generator=mask_generator, 
        mask_gen_batch_size=mask_gen_batch_size, 
    )