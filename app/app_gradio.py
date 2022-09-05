import json

import gradio as gr
import pandas as pd
import requests

from app import utils


def create_app() -> gr.Interface:
    """Initialize app.

    Returns:
        gr.Interface: gradio interface.
    """

    # Default settings
    logger = utils.create_logger()
    columns = [
        "temperature_c",
        "humidity",
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
    logger.info("Ready for inference!")

    def predict(*args):
        # Prepare input
        df = pd.DataFrame([args], columns=columns)
        df = df.astype(
            {"tvoc_ppb": int, "e_co_2_ppm": int, "raw_h_2": int, "raw_ethanol": int, "cnt": int}
        )

        # Validate input
        valid_df = utils.SmokeFeatures(**(df.to_dict(orient="records")[0]))

        # Predict
        output = requests.post(
            "http://localhost:3000/predict",
            headers={"content-type": "application/json"},
            data=json.dumps(valid_df.dict()),
        ).text

        return output["predictions"]

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
    logger.info("Smoke Detector App is ready!")
    return smokeapp


if __name__ == "__main__":
    smokeapp = create_app()
    smokeapp.launch()
