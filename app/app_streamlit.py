import argparse
from pathlib import Path

import bentoml
import pandas as pd
import shap
import streamlit as st

from app import app_utils
from config import config

st.set_option("deprecation.showPyplotGlobalUse", False)


@st.cache()
def load_data():
    projects_fp_raw = Path(config.DATA_DIR, config.DATA_RAW_NAME)
    df_raw = pd.read_csv(projects_fp_raw)

    projects_fp_fil = Path(config.DATA_DIR, config.DATA_PREPROCESS_WITHOUT_OUTLINES_NAME)
    df_clear = pd.read_csv(projects_fp_fil)
    return df_raw, df_clear


@st.cache()
def calculate_shap_values(explainer: shap.Explainer, X: pd.DataFrame):
    """_summary_

    Args:
        explainer (shap.Explainer): _description_
        X (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    return explainer(X)


def create_app(gradio_url: str = "http://localhost:8760"):
    """Create streamlit app.

    Args:
        gradio_url (str, optional): Url of gradio. Defaults to "http://localhost:8760".
    """
    # Title
    st.title("SmokeApp · Smoke detection AI")

    # Data
    df_raw, df_clear = load_data()
    st.header("🔢 Data")
    st.text(
        "The data is from the [Kaggle Smoke Detector Dataset](https://www.kaggle.com/andrewmvd/smoke-detection)."
    )
    st.text("Raw data")
    projects_fp_raw = Path(config.DATA_DIR, config.DATA_RAW_NAME)
    df_raw = pd.read_csv(projects_fp_raw)
    st.text(f"Raw projects (count: {len(df_raw)})")
    st.write(df_raw)
    st.text("Filtered data")
    projects_fp_fil = Path(config.DATA_DIR, config.DATA_PREPROCESS_WITHOUT_OUTLINES_NAME)
    df_clear = pd.read_csv(projects_fp_fil)
    st.text(f"Preprocessed projects (count: {len(df_clear)})")
    st.write(df_clear)

    # Model
    model = bentoml.sklearn.get("smoke_clf_model:latest")
    model_runner = model.to_runner()
    model_runner.init_local()
    st.header(
        f"📊 Performance of the model ({model.info.labels['Type']}): {model.tag.name} v{model.tag.version}"
    )
    st.text("Overall:")
    st.write(model.info.metadata["performance"])

    X_train, X_val, X_test, y_train, y_val, y_test = app_utils.get_data_splits(
        df_clear.drop(columns=["fire_alarm"], axis=1), df_clear["fire_alarm"]
    )

    X_val = pd.DataFrame(X_val, columns=df_clear.drop(columns=["fire_alarm"], axis=1).columns)
    X_test = pd.DataFrame(X_test, columns=df_clear.drop(columns=["fire_alarm"], axis=1).columns)

    # Explainability
    st.header("🚀 Explainability")
    explainer = shap.Explainer(model_runner.predict.run, X_val, feature_names=X_val.columns)

    shap_values = calculate_shap_values(explainer, X_val)
    st.text("Validation data")
    st.text("Feature importance")
    st.pyplot(shap.summary_plot(shap_values, plot_type="violin", show=False))
    st.pyplot(shap.plots.bar(shap_values, show=False))
    st.text("Violin plot")

    shap_values = calculate_shap_values(explainer, X_test)
    st.text("Test data")
    st.text("Feature importance")
    st.pyplot(shap.plots.bar(shap_values, show=False))
    st.text("Violin plot")
    st.pyplot(shap.summary_plot(shap_values, plot_type="violin", show=False))

    # Inference
    st.header(f"🚀 For Inference we suggest you to visit gradio page {gradio_url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create streamlit app.")
    parser.add_argument(
        "--gradio_url", type=str, default="http://localhost:8760", help="Url of gradio app."
    )
    args = parser.parse_args()
    create_app(args.gradio_url)
