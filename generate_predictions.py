import pandas as pd
import joblib
from tensorflow.keras.models import load_model

print("📥 Loading model & preprocessor...")

ann = load_model("model_artifacts/fraud_ann_model.h5")
preprocessor = joblib.load("model_artifacts/preprocessor.joblib")

print("📄 Loading data...")
df = pd.read_csv("transaction.csv")

# Ground truth exists here
y_true = df["is_fraud"]

X = df.drop(columns=["is_fraud"])
X_proc = preprocessor.transform(X)

print("🤖 Predicting...")
ann_prob = ann.predict(X_proc).ravel()

# 🔥 Threshold tuned for high accuracy
THRESHOLD = 0.85
y_pred = (ann_prob >= THRESHOLD).astype(int)

df["fraud_probability"] = ann_prob
df["fraud_prediction"] = y_pred

df.to_csv("fraud_predictions_final_v2.csv", index=False)
print("✅ Predictions saved: fraud_predictions_final_v2.csv")
