"""Public-facing Streamlit app for automobile price predictions."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="Automobile Price Predictor",
    page_icon="🚘",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load trusted model files created by train_model.py."""
    random_forest = joblib.load(
        ARTIFACT_DIR / "random_forest_pipeline.pkl"
    )
    linear_regression = joblib.load(
        ARTIFACT_DIR / "linear_regression_pipeline.pkl"
    )
    return {
        "Random Forest": random_forest,
        "Linear Regression": linear_regression
    }


@st.cache_data
def load_json(filename):
    with open(ARTIFACT_DIR / filename, "r", encoding="utf-8") as file:
        return json.load(file)


st.title("🚘 Automobile Price Predictor")
st.write(
    "Enter the vehicle specifications below to estimate its price. "
    "This is a machine-learning estimate, not a guaranteed market price."
)

required_files = [
    ARTIFACT_DIR / "random_forest_pipeline.pkl",
    ARTIFACT_DIR / "linear_regression_pipeline.pkl",
    ARTIFACT_DIR / "app_metadata.json",
    ARTIFACT_DIR / "metrics.json",
]
missing = [str(path.name) for path in required_files if not path.exists()]

if missing:
    st.error(
        "Model files are missing. Run `python train_model.py` first. "
        f"Missing files: {', '.join(missing)}"
    )
    st.stop()

models = load_models()
metadata = load_json("app_metadata.json")
metrics = load_json("metrics.json")

with st.sidebar:
    st.header("Prediction settings")
    selected_model = st.selectbox(
        "Choose a model",
        ["Random Forest", "Linear Regression"],
        index=0
    )
    st.caption(
        "Random Forest is selected by default. Compare the evaluation "
        "metrics below before deciding which model to use."
    )

st.subheader("Vehicle specifications")

feature_values = {}
numeric_columns = set(metadata["numeric_columns"])
categorical_columns = set(metadata["categorical_columns"])

with st.form("prediction_form"):
    # Divide inputs into columns to keep the form compact.
    form_columns = st.columns(3)

    for i, feature in enumerate(metadata["feature_columns"]):
        target_column = form_columns[i % len(form_columns)]

        with target_column:
            label = feature.replace("-", " ").replace("_", " ").title()

            if feature in numeric_columns:
                default = float(metadata["numeric_defaults"].get(feature, 0.0))
                step = float(metadata["numeric_steps"].get(feature, 0.1))
                feature_values[feature] = st.number_input(
                    label,
                    value=default,
                    step=step,
                    key=f"numeric_{feature}"
                )
            elif feature in categorical_columns:
                options = metadata["categorical_options"].get(feature, [])
                default = metadata["categorical_defaults"].get(feature, "")

                if options:
                    index = options.index(default) if default in options else 0
                    feature_values[feature] = st.selectbox(
                        label,
                        options=options,
                        index=index,
                        key=f"category_{feature}"
                    )
                else:
                    feature_values[feature] = st.text_input(
                        label,
                        value=default,
                        key=f"text_{feature}"
                    )

    submitted = st.form_submit_button(
        "Predict automobile price",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame(
        [[feature_values[col] for col in metadata["feature_columns"]]],
        columns=metadata["feature_columns"]
    )

    try:
        prediction = float(models[selected_model].predict(input_df)[0])
        st.subheader("Estimated price")
        st.metric(
            label=f"{selected_model} prediction",
            value=f"{prediction:,.2f}"
        )
        st.caption(
            "The displayed amount uses the same price units/currency as "
            "the training dataset. Predictions may be unreliable for cars "
            "whose specifications differ greatly from the training data."
        )
    except Exception as error:
        st.error(
            "The prediction could not be generated. Check that the saved "
            "models were created with this version of the app."
        )
        st.exception(error)

st.divider()
st.subheader("Model evaluation")
st.caption("Metrics are calculated on the held-out test split during training.")

metric_columns = st.columns(len(metrics))
for column, (name, values) in zip(metric_columns, metrics.items()):
    with column:
        st.markdown(f"**{name}**")
        st.metric("R²", f"{values['r2']:.3f}")
        st.metric("MAE", f"{values['mae']:,.2f}")
        st.metric("MSE", f"{values['mse']:,.2f}")

plot_path = ARTIFACT_DIR / "actual_vs_predicted.png"
if plot_path.exists():
    with st.expander("View actual vs. predicted chart"):
        st.image(str(plot_path), use_container_width=True)

st.caption(
    "Built with Python, pandas, scikit-learn, and Streamlit. "
    "For educational purposes; not financial or purchasing advice."
)
