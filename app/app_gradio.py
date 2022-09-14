import os
from typing import Any

import app_utils
import gradio as gr


def create_app():
    """Create gradio app.

    Returns:
        Gradio app
    """

    # Default settings
    logger = app_utils.create_logger()
    logger.info("Ready for inference!")

    # Get API url
    api_url = os.getenv("API_URL", "http://localhost:3000")

    def gradio_predict(*args: Any) -> str:
        return app_utils.predict_api(*args, api_url=api_url)

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
    smokeapp = create_app()
    smokeapp.launch(server_name="0.0.0.0", server_port=3001)
