
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "mtcars.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

FEATURES: list[str] = ["wt", "hp"]
TARGET: str = "mpg"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    y = df[TARGET].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X, y, scoring="r2", cv=cv)

    print("Model: LinearRegression")
    print(f"Features: {FEATURES}")
    print(f"Target:   {TARGET}")
    print(f"Intercept: {model.intercept_:.4f}")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"  coef[{name}] = {coef:+.4f}")
    print()
    print("In-sample metrics:")
    print(f"  R^2  = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print()
    print("5-fold CV R^2:")
    print(f"  scores = {np.round(cv_r2, 4).tolist()}")
    print(f"  mean   = {cv_r2.mean():.4f}  (std = {cv_r2.std():.4f})")

    artifact = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "metrics": {
            "r2_in_sample": r2,
            "rmse_in_sample": rmse,
            "mae_in_sample": mae,
            "r2_cv_mean": float(cv_r2.mean()),
            "r2_cv_std": float(cv_r2.std()),
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved model artifact to {MODEL_PATH}")


if __name__ == "__main__":
    main()
