# ================================
# train_model.py
# HYBRID FRAUD MODEL TRAINING
# ANN + DNN + SMOTE
# ================================

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc

from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "transaction.csv"
OUT_DIR = "model_artifacts"
MAX_ROWS = 200_000

os.makedirs(OUT_DIR, exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(CSV_PATH)

if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42)

y = df["is_fraud"].astype(int)
X = df.drop(columns=["is_fraud"])

# -----------------------------
# FEATURES
# -----------------------------
num_cols = [
    "Transaction Amount",
    "Quantity",
    "Customer Age",
    "Account Age Days",
    "Transaction Hour"
]

cat_cols = [
    "Payment Method",
    "Product Category",
    "Customer Location",
    "Device Used"
]

numeric_tf = Pipeline([
    ("scaler", StandardScaler())
])

categorical_tf = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=True, max_categories=20))
])

preprocessor = ColumnTransformer([
    ("num", numeric_tf, num_cols),
    ("cat", categorical_tf, cat_cols)
])

print("⚙ Preprocessing...")
X_sparse = preprocessor.fit_transform(X)
joblib.dump(preprocessor, f"{OUT_DIR}/preprocessor.joblib")

# Convert to dense AFTER encoding
X_dense = X_sparse.toarray()

# -----------------------------
# SMOTE
# -----------------------------
# -----------------------------
# SMOTE
# -----------------------------
print("⚖ Applying SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_dense, y)

# -----------------------------
# LDA
# -----------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

print("📉 Applying LDA...")
lda = LinearDiscriminantAnalysis()
X_res = lda.fit_transform(X_res, y_res)

joblib.dump(lda, f"{OUT_DIR}/lda.joblib")

# -----------------------------
# TRAIN TEST SPLIT (IMPORTANT)
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_res,
    y_res,
    test_size=0.2,
    stratify=y_res,
    random_state=42
)

print("Shape after LDA:", X_train.shape)

# -----------------------------
# ANN MODEL
# -----------------------------
ann = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

ann.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")]
)

print("🚀 Training ANN...")
ann.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=256,
    callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
    verbose=2
)

ann.save(f"{OUT_DIR}/fraud_ann_model.h5")

# -----------------------------
# DNN MODEL
# -----------------------------
dnn = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.35),
    Dense(128, activation="relu"),
    Dropout(0.30),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

dnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auc")]
)

print("🚀 Training DNN...")
dnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=256,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=2
)

dnn.save(f"{OUT_DIR}/fraud_dnn_model.h5")

print("✅ TRAINING COMPLETE — ANN + DNN + SMOTE")

# ================================
# EVALUATION
# ================================

print("📊 Generating Evaluation Metrics...")

y_pred_prob = dnn.predict(X_val).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

# ================================
# CONFUSION MATRIX
# ================================
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(3, 3))   # smaller figure
plt.imshow(cm, cmap="Blues")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=80)
plt.close()

# ================================
# ROC CURVE
# ================================
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(3, 3))   # smaller figure
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=80)
plt.close()

print("📊 ROC Curve & Confusion Matrix saved inside model_artifacts/")

