"""
Image Processing Script for Task 2
Extracts features (embeddings, histograms, HOG) from all member images
and saves them to image_features.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import pillow_heif
from skimage.feature import hog
from skimage.color import rgb2gray

# Get base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "image_features.csv")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Register HEIF opener for HEIC support
pillow_heif.register_heif_opener()

def load_image(path, size=(224, 224)):
    """Load and resize image"""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size)
    return img

def extract_histogram_features(img):
    """
    Extract histogram features from RGB channels
    Returns statistical features (mean, std, min, max) for each channel
    """
    img_array = np.array(img)
    features = {}
    
    for i, channel_name in enumerate(['R', 'G', 'B']):
        channel = img_array[:, :, i].flatten()
        features[f'hist_{channel_name}_mean'] = float(np.mean(channel))
        features[f'hist_{channel_name}_std'] = float(np.std(channel))
        features[f'hist_{channel_name}_min'] = float(np.min(channel))
        features[f'hist_{channel_name}_max'] = float(np.max(channel))
        features[f'hist_{channel_name}_median'] = float(np.median(channel))
    
    # Overall histogram statistics
    gray = rgb2gray(img_array)
    features['hist_gray_mean'] = float(np.mean(gray))
    features['hist_gray_std'] = float(np.std(gray))
    
    return features

def extract_hog_features(img):
    """
    Extract HOG (Histogram of Oriented Gradients) features
    FIXED: Handles all return types from hog() function
    """
    gray = rgb2gray(np.array(img))
    try:
        # Try with visualize=False first (returns only feature vector)
        hog_result = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
            transform_sqrt=True,
            feature_vector=True,
        )
        # Convert to numpy array and flatten - NO UNPACKING
        features_vector = np.array(hog_result).flatten()
    except (ValueError, TypeError) as e:
        # Fallback: use visualize=True and take first element
        try:
            hog_result = hog(
                gray,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                visualize=True,
                transform_sqrt=True,
                feature_vector=True,
            )
            # When visualize=True, returns (features, hog_image)
            if isinstance(hog_result, tuple):
                features_vector = np.array(hog_result[0]).flatten()
            else:
                features_vector = np.array(hog_result).flatten()
        except Exception as e2:
            print(f"  Warning: HOG extraction failed, using defaults: {e2}")
            # Return minimal features if HOG fails
            return {'hog_mean': 0.0, 'hog_std': 0.0, 'hog_max': 0.0, 'hog_min': 0.0}
    
    # Return HOG features as a dictionary with indices
    hog_dict = {f'hog_{i}': float(val) for i, val in enumerate(features_vector)}
    # Also add summary statistics
    hog_dict['hog_mean'] = float(np.mean(features_vector))
    hog_dict['hog_std'] = float(np.std(features_vector))
    hog_dict['hog_max'] = float(np.max(features_vector))
    hog_dict['hog_min'] = float(np.min(features_vector))
    
    return hog_dict

def extract_embedding_features(img):
    """
    Extract simple embedding-like features using image statistics
    For a more advanced approach, you could use pre-trained models like ResNet, VGG, etc.
    """
    img_array = np.array(img)
    
    # Flatten the image to create a simple embedding
    # In practice, you might use a pre-trained CNN for better embeddings
    flattened = img_array.flatten()
    
    # Use PCA-like approach: sample key features
    # Take mean of patches to create a compact representation
    h, w = img_array.shape[:2]
    patch_size = 8
    patches_h = h // patch_size
    patches_w = w // patch_size
    
    embedding = []
    for i in range(patches_h):
        for j in range(patches_w):
            patch = img_array[i*patch_size:(i+1)*patch_size, 
                              j*patch_size:(j+1)*patch_size]
            embedding.extend([
                np.mean(patch[:, :, 0]),  # R channel mean
                np.mean(patch[:, :, 1]),  # G channel mean
                np.mean(patch[:, :, 2]),  # B channel mean
            ])
    
    # Create embedding dictionary
    embedding_dict = {f'embedding_{i}': float(val) for i, val in enumerate(embedding)}
    
    # Add summary statistics
    embedding_dict['embedding_mean'] = float(np.mean(embedding))
    embedding_dict['embedding_std'] = float(np.std(embedding))
    
    return embedding_dict

def extract_all_features(image_path):
    """
    Extract all features from an image
    """
    img = load_image(image_path)
    
    # Extract different feature types
    histogram_features = extract_histogram_features(img)
    hog_features = extract_hog_features(img)
    embedding_features = extract_embedding_features(img)
    
    # Combine all features
    all_features = {**histogram_features, **hog_features, **embedding_features}
    
    return all_features

def parse_filename(filename):
    """
    Parse filename to extract member name and expression
    Handles multiple formats:
    - memberX_neutral.jpg
    - memberX-neutral.jpg
    - Honorine-neutral.HEIC
    - Charlotte Kariza Suprised Pic .jpeg
    - Emmanuel Kwizera 2.jpg
    """
    import re
    name = os.path.splitext(filename)[0]
    
    # Define expression keywords (case insensitive)
    expression_keywords = {
        'neutral': ['neutral', 'neutre'],
        'smile': ['smile', 'smiling', 'smiles', 'happy'],
        'surprised': ['surprised', 'surprise', 'surprising', 'suprised']
    }
    
    # Try to find expression keyword in filename (case insensitive)
    name_lower = name.lower()
    expression = "unknown"
    found_keyword = None
    
    for expr, keywords in expression_keywords.items():
        for keyword in keywords:
            if keyword in name_lower:
                expression = expr
                found_keyword = keyword
                break
        if found_keyword:
            break
    
    # Extract member name by removing the expression keyword
    if found_keyword:
        # Remove the expression keyword and any separators around it
        # Handle both hyphen and underscore separators
        pattern = r'[\s_-]*' + re.escape(found_keyword) + r'[\s_-]*'
        member = re.sub(pattern, '', name, flags=re.IGNORECASE).strip()
        # Clean up any remaining separators at the end
        member = re.sub(r'[\s_-]+$', '', member)
    else:
        # No expression keyword found, try to split by common separators
        # Try underscore first
        if '_' in name:
            parts = name.split('_')
            if len(parts) >= 2:
                member = '_'.join(parts[:-1])
                expression = parts[-1].lower()
            else:
                member = name
        # Try hyphen
        elif '-' in name:
            parts = name.split('-')
            if len(parts) >= 2:
                member = '-'.join(parts[:-1])
                expression = parts[-1].lower()
            else:
                member = name
        # Try space (take everything except last word if it looks like a number or expression)
        elif ' ' in name:
            parts = name.split()
            # If last part is a number, it's probably not an expression
            if len(parts) >= 2 and not parts[-1].isdigit():
                member = ' '.join(parts[:-1])
                expression = parts[-1].lower()
            else:
                member = name
        else:
            member = name
    
    # Clean up member name
    member = member.strip()
    if not member:
        member = "unknown"
    
    return member, expression

def main():
    """Main function to process all images and create CSV"""
    print(f"Processing images from: {IMAGES_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Find all image files
    image_files = []
    for root, dirs, files in os.walk(IMAGES_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif")):
                image_files.append(os.path.join(root, f))
    
    if not image_files:
        print(f"ERROR: No images found in {IMAGES_DIR}")
        print("Please add images with naming convention: memberX_neutral.jpg, memberX_smile.jpg, memberX_surprised.jpg")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    all_data = []
    for img_path in image_files:
        try:
            filename = os.path.basename(img_path)
            member, expression = parse_filename(filename)
            
            print(f"Processing: {filename} (Member: {member}, Expression: {expression})")
            
            # Extract features
            features = extract_all_features(img_path)
            
            # Add metadata
            features['filename'] = filename
            features['member'] = member
            features['expression'] = expression
            features['image_path'] = img_path
            
            all_data.append(features)
            
        except Exception as e:
            print(f"ERROR processing {img_path}: {str(e)}")
            continue
    
    if not all_data:
        print("ERROR: No images were successfully processed")
        sys.exit(1)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Reorder columns to have metadata first
    metadata_cols = ['filename', 'member', 'expression', 'image_path']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    df = df[metadata_cols + feature_cols]
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✓ Successfully processed {len(all_data)} images")
    print(f"✓ Features saved to: {OUTPUT_FILE}")
    print(f"✓ Total features per image: {len(feature_cols)}")
    print(f"\nFeature breakdown:")
    print(f"  - Histogram features: {len([c for c in feature_cols if c.startswith('hist_')])}")
    print(f"  - HOG features: {len([c for c in feature_cols if c.startswith('hog_')])}")
    print(f"  - Embedding features: {len([c for c in feature_cols if c.startswith('embedding_')])}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df[['filename', 'member', 'expression']].head())

if __name__ == "__main__":
    main()

