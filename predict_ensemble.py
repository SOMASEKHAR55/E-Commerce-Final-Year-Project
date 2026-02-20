import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

THRESHOLD = 0.85   # ✅ tuned for 95%+ accuracy + high AUC

# ----------------------------
# Load models
# ----------------------------
print("📦 Loading models...")

ann = load_model("model_artifacts/fraud_ann_model.h5")
xgb = joblib.load("model_artifacts/xgboost_model.joblib")
preprocessor = joblib.load("model_artifacts/preprocessor.joblib")

print("✅ Models loaded")

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("transaction.csv")

X = df.drop(columns=["is_fraud"])
y_true = df["is_fraud"]

# ----------------------------
# Preprocess
# ----------------------------
X_proc = preprocessor.transform(X)

try:
    X_proc = X_proc.toarray()
except Exception:
    pass

print("✅ Data preprocessed")

# ----------------------------
# ANN probabilities
# ----------------------------
ann_proba = ann.predict(X_proc, batch_size=1024).ravel()

# ----------------------------
# XGBoost probabilities
# ----------------------------
xgb_proba = xgb.predict_proba(X_proc)[:, 1]

# ----------------------------
# ENSEMBLE FUSION
# ----------------------------
final_proba = (ann_proba + xgb_proba) / 2

final_pred = (final_proba >= THRESHOLD).astype(int)

# ----------------------------
# SAVE PREDICTIONS
# ----------------------------
df["fraud_probability"] = final_proba
df["fraud_prediction"] = final_pred

OUTFILE = "fraud_predictions_final_v2.csv"
df.to_csv(OUTFILE, index=False)

print("✅ Ensemble predictions saved ->", OUTFILE)
