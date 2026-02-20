import pandas as pd

print("Loading original data...")
df_original = pd.read_csv("transaction.csv")[["Transaction ID", "is_fraud"]]

print("Loading predictions...")
df_pred = pd.read_csv("fraud_predictions.csv")

print("Merging labels...")
merged = df_pred.merge(df_original, on="Transaction ID", how="left")

merged.to_csv("fraud_predictions_final.csv", index=False)


print("✅ is_fraud column added to fraud_predictions.csv")
print("Columns now:", merged.columns.tolist())
