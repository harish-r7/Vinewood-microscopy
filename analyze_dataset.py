# analyze_dataset.py - Check final_dataset structure
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_folder_structure(base_path):
    """Analyze the folder structure of dataset"""
    print("="*60)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*60)
    print(f"\n📁 Analyzing: {base_path}\n")
    
    if not os.path.exists(base_path):
        print(f"❌ ERROR: Path does not exist: {base_path}")
        return None
    
    # Get all items in folder
    items = os.listdir(base_path)
    print(f"Contents of {base_path}:")
    for item in items:
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    
    return items

def scan_images(folder_path, recursive=True):
    """Scan for all images in a folder"""
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    images = []
    
    if recursive:
        # Search recursively
        for ext in image_extensions:
            images.extend(Path(folder_path).rglob(f"*{ext}"))
            images.extend(Path(folder_path).rglob(f"*{ext.upper()}"))
    else:
        # Only current directory
        for ext in image_extensions:
            images.extend(Path(folder_path).glob(f"*{ext}"))
            images.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    return list(set(images))  # Remove duplicates

def analyze_images(images):
    """Analyze image properties"""
    if not images:
        print("No images found!")
        return
    
    print(f"\n📸 Found {len(images)} images\n")
    
    # Analyze first 10 images (or all if less)
    num_to_analyze = min(10, len(images))
    print(f"Analyzing {num_to_analyze} sample images...\n")
    
    sizes = []
    shapes = []
    dtypes = []
    channels_list = []
    
    for i, img_path in enumerate(images[:num_to_analyze]):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                channels = img.shape[2] if len(img.shape) > 2 else 1
                
                sizes.append((w, h))
                shapes.append(img.shape)
                dtypes.append(img.dtype)
                channels_list.append(channels)
                
                print(f"  {i+1}. {img_path.name}")
                print(f"     Size: {w}×{h}, Channels: {channels}, Type: {img.dtype}")
        except Exception as e:
            print(f"  Error reading {img_path.name}: {e}")
    
    # Summary
    if sizes:
        print("\n" + "="*60)
        print("IMAGE PROPERTIES SUMMARY")
        print("="*60)
        
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        print(f"Width range:  {min(widths)} - {max(widths)} pixels")
        print(f"Height range: {min(heights)} - {max(heights)} pixels")
        print(f"Channels:     {set(channels_list)}")
        print(f"Data type:    {set(dtypes)}")
        
        # Check if all same size
        if len(set(sizes)) == 1:
            print(f"✓ All images have same size: {sizes[0][0]}×{sizes[0][1]}")
        else:
            print(f"⚠️ Images have different sizes!")
    
    return images

def analyze_masks(masks):
    """Analyze mask properties"""
    if not masks:
        print("No masks found!")
        return
    
    print(f"\n🎭 Found {len(masks)} masks\n")
    
    num_to_analyze = min(10, len(masks))
    print(f"Analyzing {num_to_analyze} sample masks...\n")
    
    unique_values = []
    binary_masks = 0
    
    for i, mask_path in enumerate(masks[:num_to_analyze]):
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                unique_vals = np.unique(mask)
                unique_values.append(unique_vals)
                
                # Check if binary
                if len(unique_vals) <= 2:
                    binary_masks += 1
                
                print(f"  {i+1}. {mask_path.name}")
                print(f"     Shape: {mask.shape}, Unique values: {unique_vals[:5]}...")
        except Exception as e:
            print(f"  Error reading {mask_path.name}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("MASK PROPERTIES SUMMARY")
    print("="*60)
    
    if unique_values:
        all_unique = set()
        for vals in unique_values:
            all_unique.update(vals)
        
        print(f"Mask value range: {min(all_unique)} - {max(all_unique)}")
        print(f"Binary masks: {binary_masks}/{num_to_analyze}")
        
        if binary_masks == num_to_analyze:
            print("✓ All sample masks are binary (0/1 or 0/255)")
        else:
            print("⚠️ Some masks are not binary - will convert during preprocessing")
    
    return masks

def check_matching(images, masks):
    """Check if images and masks match"""
    if not images or not masks:
        return
    
    print("\n" + "="*60)
    print("IMAGE-MASK MATCHING CHECK")
    print("="*60)
    
    # Get base names without extension
    img_names = {img.stem for img in images}
    mask_names = {mask.stem for mask in masks}
    
    print(f"Unique image names: {len(img_names)}")
    print(f"Unique mask names:  {len(mask_names)}")
    
    # Find matches
    common_names = img_names.intersection(mask_names)
    print(f"Matching pairs:     {len(common_names)}")
    
    # Find mismatches
    only_images = img_names - mask_names
    only_masks = mask_names - img_names
    
    if only_images:
        print(f"\n⚠️ Images without masks ({min(5, len(only_images))} shown):")
        for name in list(only_images)[:5]:
            print(f"    - {name}")
    
    if only_masks:
        print(f"\n⚠️ Masks without images ({min(5, len(only_masks))} shown):")
        for name in list(only_masks)[:5]:
            print(f"    - {name}")
    
    if not only_images and not only_masks:
        print("\n✓ Perfect match! Every image has a corresponding mask.")
    
    return common_names

def suggest_preprocessing_steps(images, masks, structure):
    """Suggest preprocessing steps based on analysis"""
    print("\n" + "="*60)
    print("PREPROCESSING RECOMMENDATIONS")
    print("="*60)
    
    suggestions = []
    
    # Check if images and masks are in separate folders
    if structure and 'images' in [s.lower() for s in structure]:
        suggestions.append("✓ Images folder found - good structure")
    else:
        suggestions.append("⚠️ Consider organizing images in 'images/' folder")
    
    if structure and 'masks' in [s.lower() for s in structure]:
        suggestions.append("✓ Masks folder found - good structure")
    else:
        suggestions.append("⚠️ Consider organizing masks in 'masks/' folder")
    
    # Check sizes
    if images:
        sample_img = cv2.imread(str(images[0]))
        if sample_img is not None:
            h, w = sample_img.shape[:2]
            if h != 256 or w != 256:
                suggestions.append(f"✓ Will resize images from {w}×{h} to 256×256")
    
    # Check masks
    if masks:
        sample_mask = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
        if sample_mask is not None:
            unique_vals = np.unique(sample_mask)
            if len(unique_vals) > 2:
                suggestions.append("✓ Will convert masks to binary (0/1)")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print("\n✅ Ready for preprocessing!")

def main():
    """Main analysis function"""
    # Update this path to match your dataset location
    DATASET_PATH = "./final_dataset"  # Change if needed
    
    print("\n🚀 Starting Dataset Analysis...\n")
    
    # Step 1: Analyze folder structure
    structure = analyze_folder_structure(DATASET_PATH)
    
    if structure is None:
        print("\n❌ Please update DATASET_PATH to correct location")
        print("   Current path:", DATASET_PATH)
        print("\n   Try one of these:")
        print("   - './final_dataset'")
        print("   - '../final_dataset'")
        print("   - 'C:/Users/haris/Desktop/vinewood_ai/final_dataset'")
        return
    
    # Step 2: Look for images in subfolders
    print("\n" + "="*60)
    print("SCANNING FOR IMAGES AND MASKS")
    print("="*60)
    
    # Try to find images folder
    images_folder = None
    masks_folder = None
    
    for item in structure:
        item_lower = item.lower()
        item_path = os.path.join(DATASET_PATH, item)
        
        if os.path.isdir(item_path):
            if 'image' in item_lower or 'img' in item_lower:
                images_folder = item_path
            elif 'mask' in item_lower or 'seg' in item_lower or 'label' in item_lower:
                masks_folder = item_path
    
    # If specific folders not found, scan root directory
    if not images_folder:
        images_folder = DATASET_PATH
        print("No specific images folder found - scanning root directory")
    
    if not masks_folder:
        masks_folder = DATASET_PATH
        print("No specific masks folder found - scanning root directory")
    
    # Step 3: Scan images
    print(f"\n📂 Images source: {images_folder}")
    images = scan_images(images_folder, recursive=True)
    images = analyze_images(images)
    
    # Step 4: Scan masks
    print(f"\n📂 Masks source: {masks_folder}")
    masks = scan_images(masks_folder, recursive=True)
    masks = analyze_masks(masks)
    
    # Step 5: Check matching
    if images and masks:
        common = check_matching(images, masks)
    
    # Step 6: Recommendations
    if images and masks:
        suggest_preprocessing_steps(images, masks, structure)
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    if images and masks and len(common) > 0:
        print(f"\n✅ Dataset ready for preprocessing!")
        print(f"   Total pairs: {len(common)}")
        print(f"   Train: {int(len(common)*0.7)} | Val: {int(len(common)*0.15)} | Test: {int(len(common)*0.15)}")
    else:
        print("\n⚠️ Issues detected. Please check:")
        print("   1. Are images and masks in the same folder?")
        print("   2. Do they have matching filenames?")
        print("   3. Are they valid image files?")

if __name__ == "__main__":
    main()