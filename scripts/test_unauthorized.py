"""
Test script for unauthorized face detection
Simulates unauthorized attempts as required by Task 6
"""

import os
import sys

# Add scripts directory to path
sys.path.append(os.path.dirname(__file__))

from facial_recognition_predict import is_authorized, predict_from_image, load_model

def test_unauthorized_detection(test_image_path):
    """
    Test if the model can detect unauthorized/unknown faces
    """
    print("="*60)
    print("TESTING UNAUTHORIZED FACE DETECTION")
    print("="*60)
    
    if not os.path.exists(test_image_path):
        print(f"ERROR: Image not found at {test_image_path}")
        return
    
    print(f"\nTesting image: {test_image_path}")
    
    # Load model to get threshold
    model, metadata = load_model()
    threshold = metadata.get('confidence_threshold', 0.3)
    
    print(f"Model confidence threshold: {threshold:.2%}")
    print(f"Authorized members: {list(model.classes_)}")
    
    # Predict
    predicted_member, confidence, all_predictions = predict_from_image(test_image_path)
    
    print(f"\nPrediction Results:")
    print(f"  Predicted Member: {predicted_member}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"\nTop 3 Predictions:")
    for member, prob in all_predictions[:3]:
        print(f"  {member}: {prob:.2%}")
    
    # Check authorization
    is_auth, final_member, conf, message, is_unknown = is_authorized(test_image_path)
    
    print(f"\n" + "-"*60)
    print("AUTHORIZATION CHECK:")
    print("-"*60)
    print(message)
    
    if is_unknown:
        print("\n✓ UNAUTHORIZED DETECTION: Model correctly identified unknown face")
        print("  → This face is not in the training data")
    elif is_auth:
        print("\n✓ AUTHORIZED: Face recognized and authorized")
    else:
        print("\n✗ UNAUTHORIZED: Face not authorized")
    
    return is_auth, is_unknown

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_unauthorized.py <image_path>")
        print("\nExample:")
        print("  python test_unauthorized.py ../images/unknown_person.jpg")
        print("\nThis script tests if the model can detect unauthorized/unknown faces")
        sys.exit(1)
    
    test_image_path = sys.argv[1]
    test_unauthorized_detection(test_image_path)

