from typing import Dict

import bentoml
import pandas as pd
from bentoml.io import JSON

from app import utils


def create_api() -> bentoml.Service:
    """Initialize api.

    Returns:
        bentoml.Service: Sevice.
    """
    # Default settings
    logger = utils.create_logger()
    why_logger = utils.initialize_why_logger()
    bento_model = bentoml.sklearn.get("smoke_clf_model:latest")
    smoke_clf_runner = bento_model.to_runner()

    # Define application
    svc = bentoml.Service("SmokeAI", runners=[smoke_clf_runner])

    logger.info("Ready for inference!")

    @svc.api(input=JSON(), output=JSON())
    def metadata(input):
        return {
            "name": bento_model.tag.name,
            "version": bento_model.tag.version,
            "Model type": bento_model.info.labels["Type"],
            "threshold": bento_model.info.metadata["threshold"],
        }

    @svc.api(input=JSON(pydantic_model=utils.SmokeFeatures), output=JSON())
    def predict(input_data: utils.SmokeFeatures) -> Dict[str, str]:
        input_df = pd.DataFrame([input_data.dict()])
        why_logger.log(input_df)
        y_prob = smoke_clf_runner.predict_proba.run(input_df.values)
        y_pred = utils.custom_predict(
            y_prob=y_prob, threshold=bento_model.info.metadata["threshold"]
        )
        why_logger.log({"class": y_pred[0]})
        output = "Smoke detected" if y_pred[0] == 1 else "Smoke not detected"
        return {"predictions": output}

    return svc


svc = create_api()
# TODO check how to close why logger
