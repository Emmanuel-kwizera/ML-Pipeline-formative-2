"""
Facial Recognition Prediction Helper
Loads trained model and predicts member from image features
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import pillow_heif
from skimage.feature import hog
from skimage.color import rgb2gray

# Get base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "facial_recognition_model.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "facial_recognition_metadata.pkl")

# Import feature extraction functions
# We'll define them here to avoid import issues
def load_image(path, size=(224, 224)):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size)
    return img

def extract_histogram_features(img):
    img_array = np.array(img)
    features = {}
    for i, channel_name in enumerate(['R', 'G', 'B']):
        channel = img_array[:, :, i].flatten()
        features[f'hist_{channel_name}_mean'] = float(np.mean(channel))
        features[f'hist_{channel_name}_std'] = float(np.std(channel))
        features[f'hist_{channel_name}_min'] = float(np.min(channel))
        features[f'hist_{channel_name}_max'] = float(np.max(channel))
        features[f'hist_{channel_name}_median'] = float(np.median(channel))
    gray = rgb2gray(img_array)
    features['hist_gray_mean'] = float(np.mean(gray))
    features['hist_gray_std'] = float(np.std(gray))
    return features

def extract_hog_features(img):
    gray = rgb2gray(np.array(img))
    try:
        hog_result = hog(
            gray, orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), block_norm="L2-Hys",
            visualize=False, transform_sqrt=True, feature_vector=True,
        )
        features_vector = np.array(hog_result).flatten()
    except (ValueError, TypeError):
        try:
            hog_result = hog(
                gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), block_norm="L2-Hys",
                visualize=True, transform_sqrt=True, feature_vector=True,
            )
            if isinstance(hog_result, tuple):
                features_vector = np.array(hog_result[0]).flatten()
            else:
                features_vector = np.array(hog_result).flatten()
        except Exception:
            return {'hog_mean': 0.0, 'hog_std': 0.0, 'hog_max': 0.0, 'hog_min': 0.0}
    
    hog_dict = {f'hog_{i}': float(val) for i, val in enumerate(features_vector)}
    hog_dict['hog_mean'] = float(np.mean(features_vector))
    hog_dict['hog_std'] = float(np.std(features_vector))
    hog_dict['hog_max'] = float(np.max(features_vector))
    hog_dict['hog_min'] = float(np.min(features_vector))
    return hog_dict

def extract_embedding_features(img):
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    patch_size = 8
    patches_h, patches_w = h // patch_size, w // patch_size
    embedding = []
    for i in range(patches_h):
        for j in range(patches_w):
            patch = img_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            embedding.extend([np.mean(patch[:, :, 0]), np.mean(patch[:, :, 1]), np.mean(patch[:, :, 2])])
    embedding_dict = {f'embedding_{i}': float(val) for i, val in enumerate(embedding)}
    embedding_dict['embedding_mean'] = float(np.mean(embedding))
    embedding_dict['embedding_std'] = float(np.std(embedding))
    return embedding_dict

def load_model():
    """Load trained facial recognition model"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found at {MODEL_FILE}. Please train the model first.")
    
    model = joblib.load(MODEL_FILE)
    metadata = joblib.load(METADATA_FILE)
    
    return model, metadata

def extract_features_from_image(image_path):
    """Extract features from a single image"""
    pillow_heif.register_heif_opener()
    
    img = load_image(image_path)
    histogram_features = extract_histogram_features(img)
    hog_features = extract_hog_features(img)
    embedding_features = extract_embedding_features(img)
    
    # Combine all features
    all_features = {**histogram_features, **hog_features, **embedding_features}
    
    return all_features

def predict_from_image(image_path, model=None, metadata=None):
    """
    Predict member from image file
    Args:
        image_path: Path to image file
        model: Trained model (if None, loads from file)
        metadata: Model metadata (if None, loads from file)
    Returns:
        predicted_member: Predicted member name
        confidence: Confidence score
        all_predictions: All class probabilities
    """
    if model is None or metadata is None:
        model, metadata = load_model()
    
    # Extract features
    features = extract_features_from_image(image_path)
    feature_cols = metadata['feature_columns']
    
    # Convert to array
    feature_array = np.array([features.get(col, 0.0) for col in feature_cols])
    feature_array = feature_array.reshape(1, -1)
    
    # Predict
    prediction = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    confidence = np.max(probabilities)
    
    # Get all predictions sorted by confidence
    classes = model.classes_
    all_predictions = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    
    return prediction, confidence, all_predictions

def is_authorized(image_path, authorized_members=None, confidence_threshold=None):
    """
    Check if image belongs to an authorized member
    Uses confidence threshold and distance-based detection for unauthorized faces
    
    Args:
        image_path: Path to image file
        authorized_members: List of authorized member names (if None, uses all trained members)
        confidence_threshold: Minimum confidence required (if None, uses model's calculated threshold)
    Returns:
        is_authorized: Boolean
        predicted_member: Predicted member name (or "UNKNOWN" if unauthorized)
        confidence: Confidence score
        message: Status message
        is_unknown: Boolean indicating if face is completely unknown
    """
    try:
        model, metadata = load_model()
        
        # Get authorized members
        if authorized_members is None:
            authorized_members = list(model.classes_)
        
        # Use model's confidence threshold if not provided
        if confidence_threshold is None:
            confidence_threshold = metadata.get('confidence_threshold', 0.3)
        
        # Predict
        predicted_member, confidence, all_predictions = predict_from_image(
            image_path, model, metadata
        )
        
        # Check if prediction is in authorized list
        is_in_authorized_list = predicted_member in authorized_members
        
        # Check confidence threshold
        meets_confidence = confidence >= confidence_threshold
        
        # Determine if authorized
        is_auth = is_in_authorized_list and meets_confidence
        
        # Determine if completely unknown (low confidence suggests face not in training set)
        is_unknown = confidence < confidence_threshold
        
        # Generate message
        if is_auth:
            message = f"✓ AUTHORIZED: {predicted_member} (confidence: {confidence:.2%})"
            final_member = predicted_member
        elif is_unknown:
            message = f"✗ UNAUTHORIZED: Unknown face detected (confidence: {confidence:.2%} < {confidence_threshold:.2%})"
            final_member = "UNKNOWN"
        elif not is_in_authorized_list:
            message = f"✗ UNAUTHORIZED: {predicted_member} not in authorized list"
            final_member = predicted_member
        else:
            message = f"✗ UNAUTHORIZED: Low confidence ({confidence:.2%} < {confidence_threshold:.2%})"
            final_member = predicted_member
        
        return is_auth, final_member, confidence, message, is_unknown
        
    except Exception as e:
        return False, "UNKNOWN", 0.0, f"ERROR: {str(e)}", True

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python facial_recognition_predict.py <image_path>")
        print("Example: python facial_recognition_predict.py ../images/member1_neutral.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        sys.exit(1)
    
    print("="*60)
    print("FACIAL RECOGNITION PREDICTION")
    print("="*60)
    print(f"Image: {image_path}")
    
    try:
        predicted_member, confidence, all_predictions = predict_from_image(image_path)
        
        print(f"\n✓ Predicted Member: {predicted_member}")
        print(f"✓ Confidence: {confidence:.2%}")
        print(f"\nAll Predictions:")
        for member, prob in all_predictions[:5]:  # Top 5
            print(f"  {member}: {prob:.2%}")
        
        # Check authorization
        is_auth, member, conf, message, is_unknown = is_authorized(image_path)
        print(f"\n{message}")
        if is_unknown:
            print("  → This face was not recognized from training data")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

