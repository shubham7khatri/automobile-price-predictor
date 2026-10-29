"""
Train and save automobile price prediction models.

Place Automobile_data.csv in this folder, then run:
    python train_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


BASE_DIR = Path(r"C:\Users\Shubham Khatri\data science yt\automobile_price_prediction_app").resolve().parent
DATA_PATH = BASE_DIR / "automobile.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_and_clean_data(path):
    """Load the CSV and convert the known numeric text columns."""
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)

    numeric_text_columns = [
        "normalized-losses", "bore", "stroke", "compression-ratio",
        "peak-rpm", "price", "horsepower"
    ]

    for col in numeric_text_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "price" not in df.columns:
        raise ValueError("The dataset must contain a 'price' column.")

    # Rows without a target cannot be used for supervised training.
    df = df.dropna(subset=["price"]).copy()

    if df.empty:
        raise ValueError("No rows with valid prices were found.")

    return df


def make_preprocessor(X):
    """Create preprocessing that is fitted only on each training split."""
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    knn_candidates = [
        "bore", "stroke", "horsepower",
        "peak-rpm", "engine-size", "compression-ratio"
    ]
    knn_columns = [col for col in knn_candidates if col in numeric_columns]
    other_numeric_columns = [
        col for col in numeric_columns if col not in knn_columns
    ]

    transformers = []

    if knn_columns:
        knn_pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ])
        transformers.append(("knn_numeric", knn_pipeline, knn_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if other_numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(
            ("other_numeric", numeric_pipeline, other_numeric_columns)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Put it in: {BASE_DIR}"
        )

    auto = load_and_clean_data(DATA_PATH)
    X = auto.drop(columns=["price"])
    y = auto["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    predictions = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("model", estimator)
        ])

        # Fit preprocessing and model together on training data only.
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results[model_name] = {
            "r2": float(r2_score(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred))
        }
        predictions[model_name] = y_pred

        safe_name = (
            "random_forest_pipeline.pkl"
            if model_name == "Random Forest"
            else "linear_regression_pipeline.pkl"
        )
        joblib.dump(pipeline, ARTIFACT_DIR / safe_name)
        print(f"\n{model_name}")
        print(f"R²:   {results[model_name]['r2']:.4f}")
        print(f"MSE:  {results[model_name]['mse']:.4f}")
        print(f"MAE:  {results[model_name]['mae']:.4f}")
        print(f"Saved: artifacts/{safe_name}")

    # Save metrics so the web app can display them.
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save input schema and sensible defaults for the Streamlit form.
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    metadata = {
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_defaults": {},
        "numeric_steps": {},
        "categorical_options": {},
        "categorical_defaults": {}
    }

    for col in numeric_columns:
        values = pd.to_numeric(X[col], errors="coerce").dropna()
        median = float(values.median()) if not values.empty else 0.0
        metadata["numeric_defaults"][col] = median

        if values.empty or np.all(np.isclose(values, np.round(values))):
            metadata["numeric_steps"][col] = 1.0
        else:
            metadata["numeric_steps"][col] = 0.1

    for col in categorical_columns:
        values = X[col].dropna().astype(str)
        options = sorted(values.unique().tolist())
        metadata["categorical_options"][col] = options
        if not values.empty:
            metadata["categorical_defaults"][col] = str(values.mode().iloc[0])
        elif options:
            metadata["categorical_defaults"][col] = options[0]
        else:
            metadata["categorical_defaults"][col] = ""

    with open(ARTIFACT_DIR / "app_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Plot actual vs predicted values for both models.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model_name in zip(axes, models.keys()):
        y_pred = predictions[model_name]
        ax.scatter(y_test, y_pred, alpha=0.7)
        lower = min(float(y_test.min()), float(np.min(y_pred)))
        upper = max(float(y_test.max()), float(np.max(y_pred)))
        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_title(model_name)
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Predicted price")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    print("\nTraining complete.")
    print(f"Artifacts saved in: {ARTIFACT_DIR}")
    print("Next: run `streamlit run app.py`")


if __name__ == "__main__":
    main()
