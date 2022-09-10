import argparse
import json
from typing import Any

import gradio as gr
import pandas as pd
import requests

from app import app_utils, schemas


def predict(*args: Any, api_url: str) -> str:
    """Predict function.

    Args:
        args (Any): Input arguments
        api_url (str): API url

    Returns:
        str: Prediction
    """
    # Prepare input
    df = pd.DataFrame([args], columns=app_utils.COLUMNS)

    # Validate input
    valid_df = schemas.SmokeFeatures(**(df.to_dict(orient="records")[0]))

    # Predict
    output = requests.post(
        f"{api_url}/predict",
        headers={"content-type": "application/json"},
        data=json.dumps(valid_df.dict()),
    ).text

    return output["predictions"]


def create_app(api_url: str = "http://localhost:3000"):
    """Create gradio app.

    Args:
        api_url (str): API url. Defaults to "http://localhost:3000".

    Returns:
        Gradio app
    """

    # Default settings
    logger = app_utils.create_logger()
    logger.info("Ready for inference!")

    def gradio_predict(*args: Any) -> str:
        return predict(*args, api_url=api_url)

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
                with gr.Row():
                    predict_btn = gr.Button(value="Predict")
                predict_btn.click(
                    gradio_predict,
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
    logger.info("Smoke Detector App is ready!")
    return smokeapp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create gradio app.")
    parser.add_argument("--api_url", type=str, default="http://localhost:3000", help="API url")
    args = parser.parse_args()
    smokeapp = create_app(args.api_url)
    smokeapp.launch()
