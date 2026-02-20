import pandas as pd

df = pd.read_csv("fraud_predictions.csv")

# Keep only high-risk fraud
top = df[df["fraud_prediction"] == 1]

# Sort by probability
top = top.sort_values("fraud_probability", ascending=False)

# Save report
top.to_csv("top_fraud_cases.csv", index=False)

print("✅ Saved top_fraud_cases.csv")
