# src/data.py
from typing import Tuple

import numpy as np
import pandas as pd
from dataprep.clean import clean_df
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from config import config


def get_outliers(df: pd.DataFrame) -> pd.Series:
    """Get outliers in data.

    Args:
        df (pd.DataFrame): DataFrame with raw data.

    Returns:
        pd.Series: Series with outliers.
    """
    # Calculate quantiles
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Get outliers
    outlines = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum(axis=1)
    return outlines


def cleaning(df: pd.DataFrame, outliers_numb: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaning the data.

    Args:
        df (pd.DataFrame): DataFrame with raw data.
        outliers_numb (int): Number of outliers in row to set row to be removed. This value should be at least be 1. Defaults to 5.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple with cleaned data and data without outliers.
    """
    # Drop not needed columns
    df_data = df.drop(columns=["Unnamed: 0", "UTC"])

    # Clean data with dataprep
    _, cleaned_df = clean_df(df_data)
    print(cleaned_df)
    cleaned_df["fire_alarm"] = cleaned_df["fire_alarm"].astype(int)

    # Clean outliers
    outlines = get_outliers(cleaned_df)
    assert outliers_numb > 0, "value must be grater then 0"
    cleaned_df_outlines = cleaned_df.drop(outlines[outlines > outliers_numb].index)

    # Save files
    cleaned_df.to_csv(config.DATA_DIR / config.DATA_PREPROCESS_NAME, index=False)
    cleaned_df_outlines.to_csv(
        config.DATA_DIR / config.DATA_PREPROCESS_WITHOUT_OUTLINES_NAME, index=False
    )

    return cleaned_df, cleaned_df_outlines


def oversample(X: pd.Series, y: np.ndarray) -> Tuple:
    """Oversample data.

    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.

    Returns:
        Tuple: oversampled data.
    """
    # Oversample (training set)
    oversampler = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversampler.fit_resample(X, y)
    return X_over, y_over


def get_data_splits(
    X: pd.DataFrame, y: np.ndarray, train_size: float = 0.7, use_oversample=True
) -> Tuple:
    """Generate balanced data splits.
    Args:
        X (pd.Series): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.
    Returns:
        Tuple: data splits as Numpy arrays.
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)

    # Oversample (training set)
    X_over, y_over = oversample(X_train, y_train)
    return X_over, X_val, X_test, y_over, y_val, y_test


def preprocess(df: pd.DataFrame, use_outlines: bool = False, **kwargs) -> Tuple:
    """Preprocess the data.

    Args:
        df (pd.DataFrame): DataFrame with raw data.
        use_outlines (bool): Use data with outlines. Defaults to False.

    Returns:
       Tuple: data splits as Numpy arrays.
    """
    if not (
        (config.DATA_DIR / config.DATA_PREPROCESS_NAME).is_file()
        and (config.DATA_DIR / config.DATA_PREPROCESS_WITHOUT_OUTLINES_NAME).is_file()
    ):  # clean data
        if "outliers_numb" in kwargs:
            cleaned_df, cleaned_df_outlines = cleaning(df, kwargs.get("outliers_numb"))
        else:
            cleaned_df, cleaned_df_outlines = cleaning(df)
        if use_outlines:
            df_preprocessed = cleaned_df
        else:
            df_preprocessed = cleaned_df_outlines
    else:
        if use_outlines:
            df_preprocessed = pd.read_csv(config.DATA_DIR / config.DATA_PREPROCESS_NAME)
        else:
            df_preprocessed = pd.read_csv(
                config.DATA_DIR / config.DATA_PREPROCESS_WITHOUT_OUTLINES_NAME
            )

    if "oversample" in kwargs:  # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(
            df_preprocessed.drop(columns=["fire_alarm"], axis=1),
            df_preprocessed["fire_alarm"],
            use_oversample=kwargs.get("oversample"),
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(
            df_preprocessed.drop(columns=["fire_alarm"], axis=1), df_preprocessed["fire_alarm"]
        )
    return X_train, X_val, X_test, y_train, y_val, y_test
