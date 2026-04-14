"""
train.py  (v3 — 2025 test set, predicting 2026)
================================================
Trains a Gradient Boosting classifier on F1 race data.

Split strategy:
  - Train : 1990–2024  (full history including 2024)
  - Test  : 2025       (complete season — Norris WDC, McLaren WCC)
  - Predict target: 2026 season

Key improvements from v2:
  - Test set is now the freshest possible season (2025)
  - 2025 driver history feeds into predict.py for 2026 predictions
  - Accuracy report shows 2025 per-race performance
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "grid_position", "grid_squared", "is_pole",
    "driver_champ_points", "team_champ_points",
    "circuit_win_rate", "circuit_podium_rate", "circuit_type_enc",
    "driver_recent_form", "driver_season_win_pct",
    "driver_season_poles_pct", "constructor_season_wins",
    "constructor_avg_grid", "dnf_rate",
]
TARGET_COL   = "won"
TRAIN_CUTOFF = 2024   # Train 1990–2023 | Test 2024–2025 | Predict 2026


def load_features() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("features.csv not found — run python src/features.py first.")
    return pd.read_csv(path)


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders = {}
    for col in ["driver_id", "constructor_id", "circuit_id"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def per_race_accuracy(test_df: pd.DataFrame) -> dict:
    top1 = top3 = total = 0
    missed = []
    for (season, rnd), race in test_df.groupby(["season", "round"]):
        winners = race[race["won"] == 1]["driver_id"].values
        if not len(winners):
            continue
        winner = winners[0]
        ranked = race.nlargest(3, "win_prob")["driver_id"].values
        if ranked[0] == winner:
            top1 += 1
        else:
            missed.append({"season": season, "round": rnd,
                           "predicted": ranked[0], "actual": winner})
        if winner in ranked:
            top3 += 1
        total += 1
    return {
        "top1": top1, "top3": top3, "total": total,
        "top1_pct": top1 / total if total else 0,
        "top3_pct": top3 / total if total else 0,
        "missed": missed,
    }


def print_accuracy_report(metrics: dict, auc: float):
    w = 56
    def bar(p, w=20): return "█" * int(p * w) + "░" * (w - int(p * w))

    t1, t3 = metrics["top1_pct"], metrics["top3_pct"]
    print()
    print("  " + "=" * w)
    print(f"  {'ACCURACY REPORT — 2025 TEST SET':^{w}}")
    print("  " + "=" * w)
    print(f"\n  Top-1  {t1:>6.1%}  {bar(t1)}  ({metrics['top1']}/{metrics['total']} races)")
    print(f"  Top-3  {t3:>6.1%}  {bar(t3)}  ({metrics['top3']}/{metrics['total']} races)")
    print(f"  AUC    {auc:>6.4f}  {bar(auc)}")

    grade = ("🟢 EXCELLENT (≥90%!)" if t3 >= 0.90 else
             "🟡 GOOD — near target" if t3 >= 0.80 else
             "🟠 FAIR")
    print(f"\n  Top-3 grade: {grade}")
    print("  " + "=" * w)

    if metrics["missed"]:
        print(f"\n  Missed predictions (2025):")
        for m in metrics["missed"][:6]:
            print(f"    Rd{m['round']:>2}  predicted={m['predicted']:<22} actual={m['actual']}")
    print()


def train_and_evaluate():
    print("=== F1 Predictor  train.py v3 ===\n")
    print("  Training: 1990–2024   |   Test: 2025   |   Predicting: 2026\n")

    df = load_features()
    df, encoders = encode_ids(df)

    available     = [c for c in FEATURE_COLS if c in df.columns]
    all_features  = available + ["driver_id_enc", "constructor_id_enc", "circuit_id_enc"]

    train_df = df[df["season"] <  TRAIN_CUTOFF].copy()
    test_df  = df[df["season"] >= TRAIN_CUTOFF].copy()   # 2024–2025

    if test_df.empty:
        print("  ⚠️  No 2025 data found in features.csv.")
        print("  Make sure fetch_data.py and features.py have been re-run with end_year=2025.\n")
        # Fall back to 2022-2024 test split so training still works
        train_df = df[df["season"] < 2022].copy()
        test_df  = df[df["season"] >= 2022].copy()
        print(f"  Falling back: train <2022, test 2022+\n")

    X_train, y_train = train_df[all_features], train_df[TARGET_COL]
    X_test,  y_test  = test_df[all_features],  test_df[TARGET_COL]

    print(f"  Features   : {len(all_features)}")
    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test rows  : {len(test_df):,}  ({test_df['season'].min()}–{test_df['season'].max()})")
    print(f"  Win rate   : train={y_train.mean():.2%}  test={y_test.mean():.2%}\n")

    print("  Training GradientBoostingClassifier (~30s)...")
    base = GradientBoostingClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        min_samples_leaf=15, subsample=0.8, max_features="sqrt",
        random_state=42,
    )
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(X_train, y_train)
    print("  Done.\n")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    print("── Classification Report ──────────────────────")
    print(classification_report(y_test, y_pred, target_names=["No win", "Win"]))

    test_df = test_df.copy()
    test_df["win_prob"] = y_prob
    metrics = per_race_accuracy(test_df)
    print_accuracy_report(metrics, auc)

    # Feature importance from inner estimator
    inner = model.calibrated_classifiers_[0].estimator
    imp = pd.DataFrame({
        "feature": all_features,
        "importance": inner.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("── Feature Importances ────────────────────────")
    print(imp.to_string(index=False))
    print()

    # Save
    model_path    = os.path.join(MODEL_DIR, "rf_model.pkl")
    encoder_path  = os.path.join(MODEL_DIR, "encoders.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_cols.pkl")

    with open(model_path,    "wb") as f: pickle.dump(model, f)
    with open(encoder_path,  "wb") as f: pickle.dump(encoders, f)
    with open(features_path, "wb") as f: pickle.dump(all_features, f)

    print(f"  Model    → {model_path}")
    print(f"  Encoders → {encoder_path}")
    print(f"  Features → {features_path}\n")
    return model, encoders, all_features


if __name__ == "__main__":
    train_and_evaluate()