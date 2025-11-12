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
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    
    # Encode labels to numeric (needed for XGBoost and better compatibility)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✓ Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return X, y, y_encoded, df, feature_cols, label_encoder

def train_facial_recognition_model(X, y, y_encoded, label_encoder, model_type='random_forest'):
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
    
    print(f"Dataset info: {n_samples} samples, {n_classes} classes")
    
    # CRITICAL: For small datasets, check if we can do a stratified split
    # Stratified split requires test set to have at least n_classes samples
    # Calculate what test size would give us
    test_size_standard = 0.2
    estimated_test_samples = max(1, int(test_size_standard * n_samples))  # At least 1
    
    # If we can't get enough test samples for stratification, use all data
    can_do_stratified_split = (estimated_test_samples >= n_classes) and (n_samples >= n_classes * 3)
    
    if not can_do_stratified_split:
        print(f"⚠ Small dataset detected ({n_samples} samples, {n_classes} classes)")
        print(f"  Estimated test samples ({estimated_test_samples}) < classes ({n_classes}) OR insufficient data")
        print("  Using all data for training and evaluation (no test split)")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        # We have enough data for a proper split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_standard, random_state=42, stratify=y
            )
        except ValueError as e:
            # If stratified split fails for any reason, use all data
            print(f"⚠ Stratified split failed: {str(e)}")
            print("  Using all data for training and evaluation")
            X_train, X_test = X, X
            y_train, y_test = y, y
    
    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Scale features for better performance (especially for Logistic Regression and XGBoost)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels for training
    y_train_encoded = label_encoder.transform(y_train) if len(set(y_train)) > 1 else y_train
    y_test_encoded = label_encoder.transform(y_test) if len(set(y_test)) > 1 else y_test
    
    # Train model based on type
    # Choose whether to use scaled features based on model type
    use_scaled = model_type in ['logistic_regression', 'xgboost']
    X_train_final = X_train_scaled if use_scaled else X_train
    X_test_final = X_test_scaled if use_scaled else X_test
    
    if model_type == 'random_forest':
        # Random Forest works well with raw features
        model = RandomForestClassifier(
            n_estimators=200,  # Increased for better performance
            max_depth=15,  # Reduced to prevent overfitting on small dataset
            min_samples_split=3,  # Reduced for small dataset
            min_samples_leaf=1,  # Reduced for small dataset
            max_features='sqrt',  # Better for high-dimensional data
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train_final, y_train_encoded)
    elif model_type == 'logistic_regression':
        # Logistic Regression benefits from scaled features
        model = LogisticRegression(
            max_iter=2000,  # Increased iterations
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            C=0.1,  # Reduced for regularization (better for small datasets)
            penalty='l2'
        )
        model.fit(X_train_final, y_train_encoded)
    elif model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
            # XGBoost needs numeric labels and benefits from scaled features
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,  # Reduced to prevent overfitting
                learning_rate=0.05,  # Reduced learning rate for better convergence
                subsample=0.8,  # Add subsampling for regularization
                colsample_bytree=0.8,  # Feature subsampling
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False  # Use native label encoding
            )
            model.fit(X_train_final, y_train_encoded)
        except ImportError:
            print("⚠ XGBoost not available, using Random Forest instead")
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_final, y_train_encoded)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nTraining {model_type} model...")
    
    # Predictions (decode back to original labels)
    y_train_pred_encoded = model.predict(X_train_final)
    y_test_pred_encoded = model.predict(X_test_final)
    
    # Decode predictions back to original labels
    y_train_pred = label_encoder.inverse_transform(y_train_pred_encoded)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)
    
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
    
    return model, scaler, {
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

def save_model(model, scaler, label_encoder, metrics, feature_cols):
    """Save the trained model and metadata"""
    model_file = os.path.join(MODEL_DIR, "facial_recognition_model.pkl")
    scaler_file = os.path.join(MODEL_DIR, "facial_recognition_scaler.pkl")
    metadata_file = os.path.join(MODEL_DIR, "facial_recognition_metadata.pkl")
    
    # Save model
    joblib.dump(model, model_file)
    print(f"\n✓ Model saved to: {model_file}")
    
    # Save scaler
    joblib.dump(scaler, scaler_file)
    print(f"✓ Scaler saved to: {scaler_file}")
    
    # Save metadata (including label encoder)
    metadata = {
        'metrics': metrics,
        'feature_columns': feature_cols,
        'model_type': metrics['model_type'],
        'label_encoder': label_encoder  # Save label encoder for predictions
    }
    joblib.dump(metadata, metadata_file)
    print(f"✓ Metadata saved to: {metadata_file}")
    
    return model_file, scaler_file, metadata_file

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
    X, y, y_encoded, df, feature_cols, label_encoder = load_and_prepare_data()
    
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
            model, scaler, metrics = train_facial_recognition_model(X, y, y_encoded, label_encoder, model_type=model_type)
            models_trained[model_type] = (model, scaler, metrics)
            
            if metrics['test_accuracy'] > best_score:
                best_score = metrics['test_accuracy']
                best_model = (model, scaler, metrics, model_type)
        except Exception as e:
            print(f"⚠ Error training {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not models_trained:
        print("ERROR: Failed to train any models")
        sys.exit(1)
    
    # Save best model
    if best_model:
        model, scaler, metrics, model_type = best_model
        print(f"\n✓ Best model: {model_type} (Test Accuracy: {metrics['test_accuracy']:.4f})")
        save_model(model, scaler, label_encoder, metrics, feature_cols)
    
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

