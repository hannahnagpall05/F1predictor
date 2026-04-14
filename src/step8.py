# step8_accuracy.py — run this to generate the Step 8 output for your assignment

import pickle
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Load saved model and features (produced by train.py)
model        = pickle.load(open("models/rf_model.pkl", "rb"))
encoders     = pickle.load(open("models/encoders.pkl", "rb"))
feature_cols = pickle.load(open("models/feature_cols.pkl", "rb"))

# Load features and build test set
df = pd.read_csv("data/features.csv")

for col, le in encoders.items():
    df[col + "_enc"] = df[col].apply(
        lambda x: int(le.transform([x])[0]) if x in le.classes_ else 0
    )

test_df = df[df["season"] >= 2024].copy()
X_test  = test_df[[c for c in feature_cols if c in test_df.columns]]
y_test  = test_df["won"]

# 8. Calculate accuracy
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC: ", roc_auc_score(y_test, y_prob))
print()
print(classification_report(y_test, y_pred, target_names=["No Win", "Win"]))
