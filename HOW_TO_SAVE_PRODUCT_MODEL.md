# How to Save the Product Recommendation Model

## 📍 **WHERE TO ADD THE CODE**

Add a **NEW CELL** after **Cell 16** (Model Comparison).

### Current Cell Structure:
- **Cell 15**: Model Training (creates `log_pipe`, `rf_pipe`, `xgb_pipe` and their metrics)
- **Cell 16**: Model Comparison (creates `results` DataFrame)
- **Cell 17**: Multimodal Logic (markdown explanation)

### ⚠️ **IMPORTANT: Fix Cell 16 First!**

Cell 16 has a bug - it uses `lr_metrics` but the variable is actually `log_metrics`. 

**Fix Cell 16:**
```python
# Compare model performances
results = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": log_metrics["accuracy"], "F1": log_metrics["f1_macro"], "Loss": log_metrics["log_loss"]},  # ← Changed lr_metrics to log_metrics
    {"Model": "Random Forest", "Accuracy": rf_metrics["accuracy"], "F1": rf_metrics["f1_macro"], "Loss": rf_metrics["log_loss"]},
    {"Model": "XGBoost", "Accuracy": xgb_metrics["accuracy"], "F1": xgb_metrics["f1_macro"], "Loss": xgb_metrics["log_loss"]},
])

# Sort by best performance
results = results.sort_values(by="F1", ascending=False)
results
```

---

## ✅ **ADD NEW CELL 18: Save Product Model**

**Insert this as a NEW CODE CELL after Cell 16:**

```python
# ============================================================================
# SAVE PRODUCT RECOMMENDATION MODEL
# ============================================================================
# Save the best model (XGBoost) and all necessary components for prediction

import joblib
import os

print("="*70)
print("SAVING PRODUCT RECOMMENDATION MODEL")
print("="*70)

# Create models directory if it doesn't exist
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
print(f"✓ Models directory: {models_dir}")

# Determine best model based on F1 score
# XGBoost has the highest F1 score (0.6500)
best_model = xgb_pipe
best_model_name = "XGBoost"
best_metrics = xgb_metrics

print(f"\n📦 Saving best model: {best_model_name}")
print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
print(f"   F1-Score: {best_metrics['f1_macro']:.4f}")
print(f"   Log Loss: {best_metrics['log_loss']:.4f}")

# Save the model pipeline (includes preprocessor + classifier)
model_file = os.path.join(models_dir, 'product_recommendation_model.pkl')
joblib.dump(best_model, model_file)
print(f"✓ Model saved: {model_file}")

# Save the label encoder (needed to decode predictions)
label_encoder_file = os.path.join(models_dir, 'product_label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_file)
print(f"✓ Label encoder saved: {label_encoder_file}")

# Save metadata
metadata = {
    'model_type': best_model_name,
    'accuracy': best_metrics['accuracy'],
    'f1_score': best_metrics['f1_macro'],
    'log_loss': best_metrics['log_loss'],
    'feature_columns': list(X.columns),
    'target_classes': label_encoder.classes_.tolist(),
    'n_classes': len(label_encoder.classes_),
    'n_features': len(X.columns)
}

metadata_file = os.path.join(models_dir, 'product_metadata.pkl')
joblib.dump(metadata, metadata_file)
print(f"✓ Metadata saved: {metadata_file}")

# Verify files were created
print(f"\n" + "="*70)
print("VERIFICATION")
print("="*70)
for file_path in [model_file, label_encoder_file, metadata_file]:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✓ {os.path.basename(file_path)}: {size / 1024:.2f} KB")
    else:
        print(f"✗ {os.path.basename(file_path)}: NOT FOUND")

print(f"\n✅ Product recommendation model saved successfully!")
print(f"   You can now use this model in the system simulation.")
```

---

## 📋 **STEP-BY-STEP INSTRUCTIONS**

1. **Run Cell 15** (Model Training) - Make sure all models are trained
2. **Fix Cell 16** - Change `lr_metrics` to `log_metrics`
3. **Run Cell 16** - Verify the results DataFrame displays correctly
4. **Insert NEW Cell 18** - Add the saving code above
5. **Run Cell 18** - Save the model

---

## 🔍 **VERIFICATION**

After running the new cell, you should see:
```
======================================================================
SAVING PRODUCT RECOMMENDATION MODEL
======================================================================
✓ Models directory: models
📦 Saving best model: XGBoost
   Accuracy: 0.6591
   F1-Score: 0.6500
   Log Loss: 1.1934
✓ Model saved: models/product_recommendation_model.pkl
✓ Label encoder saved: models/product_label_encoder.pkl
✓ Metadata saved: models/product_metadata.pkl

======================================================================
VERIFICATION
======================================================================
✓ product_recommendation_model.pkl: XX.XX KB
✓ product_label_encoder.pkl: X.XX KB
✓ product_metadata.pkl: X.XX KB

✅ Product recommendation model saved successfully!
   You can now use this model in the system simulation.
```

---

## 📁 **FILES CREATED**

After running, these files will be in the `models/` directory:
- `product_recommendation_model.pkl` - The trained XGBoost model pipeline
- `product_label_encoder.pkl` - Label encoder to decode predictions
- `product_metadata.pkl` - Model metadata and information

---

## 🚀 **NEXT STEPS**

After saving the model:
1. Create system integration script
2. Add system simulation cell
3. Test the full flow: Face → Voice → Product

