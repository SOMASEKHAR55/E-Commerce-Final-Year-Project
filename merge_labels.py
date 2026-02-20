import pandas as pd

print("📥 Loading original dataset...")
df_original = pd.read_csv("transaction.csv")

print("📥 Loading predictions...")
df_pred = pd.read_csv("fraud_predictions_final_v2.csv")

# Ensure exact column name
TRANSACTION_COL = "Transaction ID"
TARGET_COL = "is_fraud"

print("🔗 Merging on Transaction ID...")
df_merged = df_pred.merge(
    df_original[[TRANSACTION_COL, TARGET_COL]],
    on=TRANSACTION_COL,
    how="left"
)

# Save final evaluation-ready file
OUT_FILE = "fraud_predictions_with_labels.csv"
df_merged.to_csv(OUT_FILE, index=False)

print("✅ Merged file saved:", OUT_FILE)
print("Columns now:", df_merged.columns.tolist())
