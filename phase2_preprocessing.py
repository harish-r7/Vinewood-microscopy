# phase2_preprocessing.py - Updated for nested folders
import os
import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

# ========== CONFIGURATION ==========
DATASET_PATH = "./final_dataset"
OUTPUT_PATH = "./data_prepared"

# Augmentation settings (Train only)
AUGMENTATION = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation': 15,
    'brightness': 0.2,
    'contrast': 0.2
}

# ========== HELPER FUNCTIONS ==========
def normalize_image(image):
    """Normalize image to [0,1]"""
    return image / 255.0

def make_binary_mask(mask):
    """Convert multi-class mask to binary (0/1)"""
    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask

def horizontal_flip(image, mask, prob=0.5):
    if random.random() < prob:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask

def vertical_flip(image, mask, prob=0.3):
    if random.random() < prob:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask

def rotate(image, mask, limit=15):
    if limit > 0 and random.random() < 0.5:
        angle = random.uniform(-limit, limit)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))
    return image, mask

def adjust_brightness_contrast(image, mask):
    if random.random() < 0.3:
        brightness = 1 + random.uniform(-AUGMENTATION['brightness'], AUGMENTATION['brightness'])
        contrast = 1 + random.uniform(-AUGMENTATION['contrast'], AUGMENTATION['contrast'])
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 50)
    return image, mask

def apply_augmentation(image, mask):
    """Apply all augmentations for training"""
    image, mask = horizontal_flip(image, mask)
    image, mask = vertical_flip(image, mask)
    image, mask = rotate(image, mask, AUGMENTATION['rotation'])
    image, mask = adjust_brightness_contrast(image, mask)
    return image, mask

def find_all_images(root_dir):
    """Find all images in directory and subdirectories"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = []
    
    for ext in image_extensions:
        images.extend(Path(root_dir).rglob(f"*{ext}"))
        images.extend(Path(root_dir).rglob(f"*{ext.upper()}"))
    
    return list(set(images))

def setup_output_dirs():
    """Create output directories"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_PATH, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, split, 'masks'), exist_ok=True)
    print(f"✓ Output directories created in: {OUTPUT_PATH}")

def process_split(input_dir, output_dir, split_name, apply_aug=False):
    """Process a dataset split (train/val/test)"""
    split_path = os.path.join(input_dir, split_name)
    
    if not os.path.exists(split_path):
        print(f"⚠️ Split folder not found: {split_path}")
        return 0
    
    # Find all images in split folder and subfolders
    all_files = find_all_images(split_path)
    
    if not all_files:
        print(f"⚠️ No images found in {split_path}")
        return 0
    
    # Group by filename (assuming image and mask have same name)
    # Filter for image files (RGB) vs mask files (grayscale)
    # We'll assume all files are pairs - each file has corresponding mask with same name
    image_files = []
    mask_files = []
    
    for file_path in all_files:
        # Check if it's likely an image (RGB) or mask (grayscale)
        # For now, we'll treat all as pairs and use same filename for both
        # The actual distinction will be made when loading
        image_files.append(file_path)
    
    # Since images and masks have the same names, we'll process each file as a pair
    # We need to find corresponding mask in the same folder
    processed_count = 0
    
    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        try:
            # Look for mask with same name in the same directory
            mask_path = img_path.parent / img_path.name
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load mask (same filename)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Try different extension for mask
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = img_path.parent / f"{img_path.stem}{ext}"
                    if test_path.exists():
                        mask = cv2.imread(str(test_path), cv2.IMREAD_GRAYSCALE)
                        break
                if mask is None:
                    continue
            
            # Preprocess (already 256x256)
            image = normalize_image(image)
            mask = make_binary_mask(mask)
            
            # Convert to uint8 for augmentation
            image_uint8 = (image * 255).astype(np.uint8)
            mask_uint8 = mask.astype(np.uint8)
            
            # Apply augmentation only for training
            if apply_aug:
                image_uint8, mask_uint8 = apply_augmentation(image_uint8, mask_uint8)
                image = image_uint8 / 255.0
                mask = mask_uint8
            
            # Save processed files
            output_img_path = os.path.join(output_dir, split_name, 'images', img_path.name)
            output_mask_path = os.path.join(output_dir, split_name, 'masks', img_path.name)
            
            save_img = (image * 255).astype(np.uint8) if not apply_aug else image_uint8
            save_mask = (mask * 255).astype(np.uint8)
            
            cv2.imwrite(output_img_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(output_mask_path, save_mask)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    return processed_count

def inspect_dataset_structure():
    """First, let's see what's inside train/val/test folders"""
    print("\n🔍 Inspecting dataset structure...")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(DATASET_PATH, split)
        if os.path.exists(split_path):
            print(f"\n📁 {split.upper()}: {split_path}")
            
            # List contents
            items = os.listdir(split_path)
            print(f"   Contents: {items[:5]}")  # Show first 5 items
            
            # Check if there are subfolders
            for item in items[:3]:
                item_path = os.path.join(split_path, item)
                if os.path.isdir(item_path):
                    sub_items = os.listdir(item_path)
                    print(f"   📂 {item}/ contains {len(sub_items)} files")
                    if sub_items:
                        print(f"      Sample: {sub_items[0]}")
        else:
            print(f"\n❌ {split.upper()} folder not found: {split_path}")

def verify_preprocessing():
    """Verify the preprocessing results"""
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(OUTPUT_PATH, split, 'images')
        mask_dir = os.path.join(OUTPUT_PATH, split, 'masks')
        
        if not os.path.exists(img_dir):
            print(f"❌ {split}: Directory not found")
            continue
        
        images = list(Path(img_dir).glob("*.png")) + list(Path(img_dir).glob("*.jpg"))
        masks = list(Path(mask_dir).glob("*.png")) + list(Path(mask_dir).glob("*.jpg"))
        
        print(f"\n📁 {split.upper()}: {len(images)} images, {len(masks)} masks")
        
        if len(images) > 0:
            # Check first sample
            img = cv2.imread(str(images[0]))
            mask = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                img_normalized = img / 255.0
                print(f"  ✓ Image shape: {img.shape}, range: [{img_normalized.min():.2f}, {img_normalized.max():.2f}]")
            
            if mask is not None:
                unique_vals = np.unique(mask)
                print(f"  ✓ Mask shape: {mask.shape}, unique values: {unique_vals}")
    
    print("\n✅ Verification complete!")

def print_summary(train_count, val_count, test_count):
    """Print processing summary"""
    print("\n" + "="*60)
    print("✅ PHASE 2: PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\n📊 Dataset Summary:")
    print(f"   Train: {train_count} images (with augmentation)")
    print(f"   Val:   {val_count} images")
    print(f"   Test:  {test_count} images")
    print(f"\n📁 Output saved to: {OUTPUT_PATH}")
    print(f"\n📋 Preprocessing applied:")
    print(f"   ✓ Images normalized to [0,1]")
    print(f"   ✓ Masks converted to binary (0/1)")
    print(f"   ✓ Train: Augmentations applied")
    print(f"   ✓ Val/Test: No augmentations")

def main():
    """Main execution"""
    print("="*60)
    print("PHASE 2: PREPROCESSING PIPELINE")
    print("="*60)
    
    # First inspect the dataset structure
    inspect_dataset_structure()
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"\n❌ ERROR: Dataset not found at {DATASET_PATH}")
        return
    
    # Setup output directories
    setup_output_dirs()
    
    print("\n🔄 Starting preprocessing...\n")
    
    # Process each split
    train_count = process_split(DATASET_PATH, OUTPUT_PATH, 'train', apply_aug=True)
    val_count = process_split(DATASET_PATH, OUTPUT_PATH, 'val', apply_aug=False)
    test_count = process_split(DATASET_PATH, OUTPUT_PATH, 'test', apply_aug=False)
    
    # Print summary
    print_summary(train_count, val_count, test_count)
    
    # Verify results
    verify_preprocessing()
    
    if train_count == 0 and val_count == 0 and test_count == 0:
        print("\n⚠️ No files were processed. Please check:")
        print("   1. Are there images inside train/val/test folders?")
        print("   2. Do the images have common extensions (.png, .jpg)?")
        print("\n💡 Run this command to see what's inside:")
        print("   dir .\\final_dataset\\train\\")

if __name__ == "__main__":
    main()