import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

print("📥 Loading predictions...")
df = pd.read_csv("fraud_predictions_final_v2.csv")

# ✅ Labels exist
y_true = df["is_fraud"]
y_pred = df["fraud_prediction"]
y_prob = df["fraud_probability"]

print("\n===== MODEL PERFORMANCE (ANN) =====")
print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(precision_score(y_true, y_pred), 4))
print("Recall   :", round(recall_score(y_true, y_pred), 4))
print("F1 Score :", round(f1_score(y_true, y_pred), 4))
print("AUC      :", round(roc_auc_score(y_true, y_prob), 4))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.savefig("roc_curve.png")
plt.close()

print("✅ Evaluation completed")
