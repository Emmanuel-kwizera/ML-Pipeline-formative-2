"""
Facial Recognition Model for Task 4
Trains a model to recognize/identify members from image features
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Get base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURES_FILE = os.path.join(BASE_DIR, "data", "processed", "image_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load image features and prepare for training"""
    print("="*60)
    print("LOADING IMAGE FEATURES")
    print("="*60)
    
    if not os.path.exists(FEATURES_FILE):
        print(f"ERROR: Features file not found at {FEATURES_FILE}")
        print("Please run image_processing.py first to generate image_features.csv")
        sys.exit(1)
    
    df = pd.read_csv(FEATURES_FILE)
    print(f"✓ Loaded {len(df)} images")
    print(f"✓ Features shape: {df.shape}")
    
    # Check for required columns
    if 'member' not in df.columns:
        print("ERROR: 'member' column not found in features file")
        sys.exit(1)
    
    # Get feature columns (exclude metadata)
    metadata_cols = ['filename', 'member', 'expression', 'image_path']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"✓ Using {len(feature_cols)} features per image")
    print(f"✓ Members in dataset: {df['member'].nunique()}")
    print(f"  Members: {sorted(df['member'].unique())}")
    
    # Prepare features and labels
    X = df[feature_cols].values
    y = df['member'].values
    
    # Handle any NaN values
    if pd.isna(X).any():
        print("⚠ Warning: NaN values found in features. Filling with 0...")
        X = np.nan_to_num(X, nan=0.0)
    
    return X, y, df, feature_cols

def train_facial_recognition_model(X, y, model_type='random_forest'):
    """
    Train facial recognition model
    Args:
        X: Feature matrix
        y: Member labels
        model_type: 'random_forest', 'logistic_regression', or 'xgboost'
    """
    print("\n" + "="*60)
    print(f"TRAINING FACIAL RECOGNITION MODEL ({model_type.upper()})")
    print("="*60)
    
    # Split data - handle small datasets
    n_samples = len(X)
    n_classes = len(np.unique(y))
    
    # For small datasets, use smaller test size or skip split
    if n_samples < 20:
        # Use smaller test size for small datasets
        # Ensure test set has at least as many samples as classes
        min_test_size = max(2, n_classes)  # At least 2 samples or number of classes
        test_size = min(0.15, min_test_size / n_samples)  # Max 15% or minimum needed
        
        # If still too small, use all data for training (no test split)
        if test_size * n_samples < n_classes:
            print(f"⚠ Very small dataset ({n_samples} samples, {n_classes} classes)")
            print("  Using all data for training (no test split)")
            X_train, X_test = X, X
            y_train, y_test = y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
    else:
        # Standard split for larger datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Train model based on type
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            C=1.0
        )
    elif model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss'
            )
        except ImportError:
            print("⚠ XGBoost not available, using Random Forest instead")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nTraining {model_type} model...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Calculate loss (for classification, we use log loss)
    try:
        from sklearn.metrics import log_loss
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)
        train_loss = log_loss(y_train, y_train_proba)
        test_loss = log_loss(y_test, y_test_proba)
    except:
        train_loss = None
        test_loss = None
    
    # Print results
    print("\n" + "-"*60)
    print("MODEL PERFORMANCE METRICS")
    print("-"*60)
    print(f"Training Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:      {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Training F1-Score:  {train_f1:.4f}")
    print(f"Test F1-Score:      {test_f1:.4f}")
    if train_loss is not None:
        print(f"Training Loss:       {train_loss:.4f}")
        print(f"Test Loss:          {test_loss:.4f}")
    
    # Classification report
    print("\n" + "-"*60)
    print("DETAILED CLASSIFICATION REPORT (Test Set)")
    print("-"*60)
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("\n" + "-"*60)
    print("CONFUSION MATRIX (Test Set)")
    print("-"*60)
    cm = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm, 
                         index=sorted(set(y_test)),
                         columns=sorted(set(y_test)))
    print(cm_df)
    
    # Calculate confidence thresholds for unauthorized detection
    # Use training data to determine minimum acceptable confidence
    y_train_proba = model.predict_proba(X_train)
    train_confidences = np.max(y_train_proba, axis=1)
    
    # Set threshold as 2 standard deviations below mean (captures ~95% of authorized)
    confidence_threshold = np.mean(train_confidences) - 2 * np.std(train_confidences)
    # Ensure threshold is reasonable (at least 0.1)
    confidence_threshold = max(0.1, min(0.5, confidence_threshold))
    
    print(f"\nConfidence Threshold for Authorization: {confidence_threshold:.4f}")
    print(f"  (Mean training confidence: {np.mean(train_confidences):.4f})")
    print(f"  (Std training confidence: {np.std(train_confidences):.4f})")
    
    return model, {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'model_type': model_type,
        'confidence_threshold': confidence_threshold,
        'mean_train_confidence': float(np.mean(train_confidences)),
        'std_train_confidence': float(np.std(train_confidences))
    }

def save_model(model, metrics, feature_cols):
    """Save the trained model and metadata"""
    model_file = os.path.join(MODEL_DIR, "facial_recognition_model.pkl")
    metadata_file = os.path.join(MODEL_DIR, "facial_recognition_metadata.pkl")
    
    # Save model
    joblib.dump(model, model_file)
    print(f"\n✓ Model saved to: {model_file}")
    
    # Save metadata
    metadata = {
        'metrics': metrics,
        'feature_columns': feature_cols,
        'model_type': metrics['model_type']
    }
    joblib.dump(metadata, metadata_file)
    print(f"✓ Metadata saved to: {metadata_file}")
    
    return model_file, metadata_file

def predict_member(image_features, model, feature_cols):
    """
    Predict member from image features
    Args:
        image_features: Dictionary of features or array
        model: Trained model
        feature_cols: List of feature column names
    Returns:
        predicted_member: Predicted member name
        confidence: Confidence score
    """
    # Convert to array if dictionary
    if isinstance(image_features, dict):
        feature_array = np.array([image_features.get(col, 0.0) for col in feature_cols])
    else:
        feature_array = np.array(image_features)
    
    # Reshape if needed
    if len(feature_array.shape) == 1:
        feature_array = feature_array.reshape(1, -1)
    
    # Predict
    prediction = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    confidence = np.max(probabilities)
    
    return prediction, confidence

def main():
    """Main function"""
    print("="*60)
    print("FACIAL RECOGNITION MODEL TRAINING")
    print("="*60)
    
    # Load data
    X, y, df, feature_cols = load_and_prepare_data()
    
    # Check if we have enough data
    if len(df) < 10:
        print("⚠ Warning: Very few images. Model performance may be poor.")
    
    if df['member'].nunique() < 2:
        print("ERROR: Need at least 2 different members for classification")
        sys.exit(1)
    
    # Train models (try multiple)
    models_trained = {}
    best_model = None
    best_score = 0
    
    model_types = ['random_forest', 'logistic_regression']
    
    # Try XGBoost if available
    try:
        import xgboost
        model_types.append('xgboost')
    except:
        pass
    
    for model_type in model_types:
        try:
            model, metrics = train_facial_recognition_model(X, y, model_type=model_type)
            models_trained[model_type] = (model, metrics)
            
            if metrics['test_accuracy'] > best_score:
                best_score = metrics['test_accuracy']
                best_model = (model, metrics, model_type)
        except Exception as e:
            print(f"⚠ Error training {model_type}: {str(e)}")
            continue
    
    if not models_trained:
        print("ERROR: Failed to train any models")
        sys.exit(1)
    
    # Save best model
    if best_model:
        model, metrics, model_type = best_model
        print(f"\n✓ Best model: {model_type} (Test Accuracy: {metrics['test_accuracy']:.4f})")
        save_model(model, metrics, feature_cols)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Trained {len(models_trained)} model(s)")
    print(f"✓ Best model saved: {model_type}")
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ Test F1-Score: {metrics['test_f1']:.4f}")
    if metrics['test_loss']:
        print(f"✓ Test Loss: {metrics['test_loss']:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    main()

