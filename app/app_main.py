from typing import Any, Tuple

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from app import app_utils


def create_app() -> Tuple[gr.Interface, Any]:
    """Initialize app.

    Returns:
        gr.Interface: gradio interface.
    """

    # Default settings
    matplotlib.use("Agg")
    logger = app_utils.create_logger()
    why_logger = app_utils.initialize_why_logger()
    columns = [
        "temperature_c",
        "humidity_%",
        "tvoc_ppb",
        "e_co_2_ppm",
        "raw_h_2",
        "raw_ethanol",
        "pressure_h_pa",
        "pm_1_0",
        "pm_2_5",
        "nc_0_5",
        "nc_1_0",
        "nc_2_5",
        "cnt",
    ]

    # Load default artifacts
    artifacts = app_utils.load_artifacts(run_id=None)
    logger.info("Ready for inference!")

    def predict(*args):
        # Prepare input
        input_data = np.array([args])
        df = pd.DataFrame(input_data, columns=columns)

        # Log input vector as dictionary
        why_logger.log(df)

        # Predict
        y_prob = artifacts["model"].predict_proba(input_data)
        y_pred = [np.argmax(p) if max(p) > artifacts["args"].threshold else 0 for p in y_prob]
        output = "Smoke detected" if y_pred[0] == 1 else "Smoke not detected"

        # Log predicted class
        why_logger.log({"class": y_pred[0]})
        return output

    def interpret(*args):
        input_data = np.array([args])
        explainer = shap.KernelExplainer(artifacts["model"].predict, input_data)
        shap_values = explainer.shap_values(input_data)
        scores_desc = list(zip(shap_values[0], columns))
        scores_desc = sorted(scores_desc)
        fig_m = plt.figure(tight_layout=True)
        plt.barh([s[1] for s in scores_desc], [s[0] for s in scores_desc])
        plt.title("Feature Shap Values")
        plt.ylabel("Shap Value")
        plt.xlabel("Feature")
        plt.tight_layout()
        return fig_m

    with gr.Blocks() as smokeapp:
        gr.Markdown("**Smoke Detector Classification**")
        with gr.Row():
            with gr.Column():
                temperature_c = gr.Number(
                    label="Temperature (C)", min=-50, max=50, step=0.1, value=20, precison=1
                )
                humidity = gr.Number(
                    label="Humidity (%)", min=0, max=100, step=0.1, value=50, precison=1
                )
                tvoc_ppb = gr.Number(
                    label="TVOC (ppb)", min=0, max=100000, step=1, value=20000, precison=0
                )
                e_co_2_ppm = gr.Number(
                    label="eCO2 (ppm)", min=0, max=100000, step=1, value=20000, precison=0
                )
                raw_h_2 = gr.Number(
                    label="Raw H2 (ppb)", min=0, max=40000, step=1, value=10000, precison=0
                )
                raw_etanol = gr.Number(
                    label="Raw Etanol (ppb)", min=0, max=40000, step=1, value=10000, precison=0
                )
                pressure_h_pa = gr.Number(
                    label="Pressure (hPa)", min=0, max=100000, step=0.1, value=20000, precison=1
                )
                pm_1_0 = gr.Number(
                    label="PM 1.0 (ug/m3)", min=0, max=100000, step=0.1, value=20000, precison=1
                )
                pm_2_5 = gr.Number(
                    label="PM 2.5 (ug/m3)", min=0, max=100000, step=0.1, value=20000, precison=1
                )
                nc_0_5 = gr.Number(
                    label="NC 0.5 (ug/m3)", min=0, max=100000, step=0.1, value=20000, precison=1
                )
                nc_1_0 = gr.Number(
                    label="NC 1.0 (ug/m3)", min=0, max=100000, step=0.1, value=20000, precison=1
                )
                nc_2_5 = gr.Number(
                    label="NC 2.5 (ug/m3)", min=0, max=100000, step=0.1, value=20000, precison=1
                )
                cnt = gr.Number(label="CNT (ug/m3)", min=0, max=1000, step=1, value=100, precison=0)
            with gr.Column():
                label = gr.Label()
                plot = gr.Plot()
                with gr.Row():
                    predict_btn = gr.Button(value="Predict")
                    interpret_btn = gr.Button(value="Explain")
                predict_btn.click(
                    predict,
                    inputs=[
                        temperature_c,
                        humidity,
                        tvoc_ppb,
                        e_co_2_ppm,
                        raw_h_2,
                        raw_etanol,
                        pressure_h_pa,
                        pm_1_0,
                        pm_2_5,
                        nc_0_5,
                        nc_1_0,
                        nc_2_5,
                        cnt,
                    ],
                    outputs=[label],
                )
                interpret_btn.click(
                    interpret,
                    inputs=[
                        temperature_c,
                        humidity,
                        tvoc_ppb,
                        e_co_2_ppm,
                        raw_h_2,
                        raw_etanol,
                        pressure_h_pa,
                        pm_1_0,
                        pm_2_5,
                        nc_0_5,
                        nc_1_0,
                        nc_2_5,
                        cnt,
                    ],
                    outputs=[plot],
                )

    logger.info("Smoke Detector App is ready!")
    return smokeapp, why_logger


if __name__ == "__main__":
    smokeapp, why_logger = create_app()
    smokeapp.launch()
    why_logger.close()
