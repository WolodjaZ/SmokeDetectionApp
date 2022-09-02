import json
import logging
from argparse import Namespace
from typing import Dict, Optional

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import data, evaluate, predict, utils


def train(
    args: Namespace, df: pd.DataFrame, logger: logging.Logger, trial: Optional[int] = None
) -> Dict:
    """Train model on data.

    Args:
        args (Namespace): Arguments to use for training.
        df (pd.DataFrame): DataFrame with data.
        logger (logging.Logger): Logger.
        trial (Optional[int], optional): Trial number. Defaults to None.

    Raises:
        optuna.TrialPruned: early stopping of trial if it's performing poorly.

    Returns:
        dict: Dictionary with results.
    """
    # Setup
    utils.set_seeds()
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.subset]  # None = all samples
    X_train, X_val, X_test, y_train, y_val, y_test = data.preprocess(
        df, use_outlines=args.use_outlines, **vars(args)
    )
    logger.info("✅ Data preprocessed")

    # Creating pipeline
    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", GaussianNB(var_smoothing=args.var_smoothing)),
        ]
    )

    # Train data
    logger.info("✅ Model training")
    for epoch in range(args.num_epochs):
        pipe.fit(X_train, y_train)

        # Get train metrics
        train_loss = log_loss(y_train, pipe.predict_proba(X_train))
        val_loss = log_loss(y_val, pipe.predict_proba(X_val))
        logger.info(
            f"Epoch: {epoch:02d} | " f"train_loss: {train_loss:.5f}, " f"val_loss: {val_loss:.5f}"
        )
        # Log mlflow
        if not trial:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # Pruning (for optimization in next section)
        if trial:  # pragma: no cover, optuna pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    logger.info("✅ Model finished training")

    # Threshold
    y_pred = pipe.predict(X_val)
    y_prob = pipe.predict_proba(X_val)
    args.threshold = np.quantile([y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)  # Q1

    # Evaluation
    y_prob = pipe.predict_proba(X_test)
    y_pred = predict.custom_predict(y_prob=y_prob, threshold=args.threshold)
    performance = evaluate.get_metrics(y_true=y_test, y_pred=y_pred)
    logger.info("✅ Model evaluated")

    return {
        "args": args,
        "model": pipe,
        "performance": performance,
    }


def objective(args: Namespace, df: pd.DataFrame, logger: logging.Logger, trial: int) -> float:
    """Objective function to optimize.

    Args:
        args (Namespace): Arguments.
        df (pd.DataFrame): DataFrame with data.
        logger (logging.Logger): Logger.
        trial (int): Trial number.

    Returns:
        float: Loss.
    """
    # Parameters to tune
    args.var_smoothing = trial.suggest_loguniform("var_smoothing", 1e-5, 1e-12)
    args.outliers_numb = trial.suggest_int("outliers_numb", 1, 12)

    # Train & evaluate
    artifacts = train(args=args, df=df, logger=logger, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]
    logger.info(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])
    trial.set_user_attr("accuracy", overall_performance["balanced_accuracy"])

    return overall_performance["f1"]
