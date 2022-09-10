from typing import Any, Dict

import bentoml
import pandas as pd
from bentoml.io import JSON

from app import app_utils, schemas


def create_api() -> bentoml.Service:
    """Initialize api.

    Returns:
        bentoml.Service: Sevice.
    """
    # Default settings
    logger = app_utils.create_logger()
    why_logger = app_utils.initialize_why_logger()
    bento_model = bentoml.sklearn.get("smoke_clf_model:latest")
    smoke_clf_runner = bento_model.to_runner()

    # Define application
    svc = bentoml.Service("SmokeAI", runners=[smoke_clf_runner])
    logger.info("Ready for inference!")

    @svc.api(input=JSON(), output=JSON())
    def metadata(input: Any) -> Dict[str, Any]:
        """Get metadata endpoint.

        Args:
            input (Any): Input not used.

        Returns:
            Dict[str, Any]: Metadata.
        """
        return {
            "name": bento_model.tag.name,
            "version": bento_model.tag.version,
            "Model type": bento_model.info.labels["Type"],
            "threshold": bento_model.info.metadata["threshold"],
        }

    @svc.api(input=JSON(pydantic_model=schemas.SmokeFeatures), output=JSON())
    def predict(input_data: schemas.SmokeFeatures) -> Dict[str, str]:
        """Predict endpoint.

        Args:
            input_data (schemas.SmokeFeatures): Input data.

        Returns:
            Dict[str, str]: Prediction.
        """
        input_df = pd.DataFrame([input_data.dict()])
        why_logger.log(input_df)
        y_prob = smoke_clf_runner.predict_proba.run(input_df.values)
        y_pred = app_utils.custom_predict(
            y_prob=y_prob, threshold=bento_model.info.metadata["threshold"]
        )
        why_logger.log({"class": y_pred[0]})
        output = "Smoke detected" if y_pred[0] == 1 else "Smoke not detected"
        return {"predictions": output}

    return svc


svc = create_api()
# TODO check how to close why logger
