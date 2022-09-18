import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data
from src.config import Dataset as DatasetConfig
from src.config import Model as ModelConfig
from src.config import Path as PathConfig
from src.config import SmokeConfig


@pytest.fixture()
def df():
    data = [
        {
            "Unnamed: 0": "1",
            "UTC": "1",
            "Temperature[C]": "1",
            "Humidity[%]": "1",
            "TVOC[ppb]": "1",
            "eCO2[ppm]": "1",
            "Raw H2": "1",
            "Raw Ethanol": "1",
            "Pressure[hPa]": "1",
            "PM1.0": "1",
            "PM2.5": "1",
            "NC0.5": "1",
            "NC1.0": "1",
            "NC2.5": "1",
            "CNT": "1",
            "Fire Alarm": "1",
        },
        {
            "Unnamed: 0": "2",
            "UTC": "2",
            "Temperature[C]": "2",
            "Humidity[%]": "2",
            "TVOC[ppb]": "2",
            "eCO2[ppm]": "2",
            "Raw H2": "2",
            "Raw Ethanol": "2",
            "Pressure[hPa]": "2",
            "PM1.0": "2",
            "PM2.5": "2",
            "NC0.5": "2",
            "NC1.0": "2",
            "NC2.5": "2",
            "CNT": "2",
            "Fire Alarm": "2",
        },
        {
            "Unnamed: 0": "3",
            "UTC": "3",
            "Temperature[C]": "3",
            "Humidity[%]": "3",
            "TVOC[ppb]": "3",
            "eCO2[ppm]": "3",
            "Raw H2": "3",
            "Raw Ethanol": "3",
            "Pressure[hPa]": "3",
            "PM1.0": "3",
            "PM2.5": "3",
            "NC0.5": "3",
            "NC1.0": "3",
            "NC2.5": "3",
            "CNT": "3",
            "Fire Alarm": "1",
        },
        {
            "Unnamed: 0": "4",
            "UTC": "4",
            "Temperature[C]": "4",
            "Humidity[%]": "4",
            "TVOC[ppb]": "4",
            "eCO2[ppm]": "4",
            "Raw H2": "4",
            "Raw Ethanol": "4",
            "Pressure[hPa]": "4",
            "PM1.0": "4",
            "PM2.5": "4",
            "NC0.5": "4",
            "NC1.0": "4",
            "NC2.5": "4",
            "CNT": "4",
            "Fire Alarm": "2",
        },
        {
            "Unnamed: 0": "5",
            "UTC": "5",
            "Temperature[C]": "5",
            "Humidity[%]": "5",
            "TVOC[ppb]": "5",
            "eCO2[ppm]": "5",
            "Raw H2": "5",
            "Raw Ethanol": "5",
            "Pressure[hPa]": "5",
            "PM1.0": "5",
            "PM2.5": "5",
            "NC0.5": "5",
            "NC1.0": "5",
            "NC2.5": "5",
            "CNT": "5",
            "Fire Alarm": "1",
        },
        {
            "Unnamed: 0": "6",
            "UTC": "6",
            "Temperature[C]": "6",
            "Humidity[%]": "6",
            "TVOC[ppb]": "6",
            "eCO2[ppm]": "6",
            "Raw H2": "6",
            "Raw Ethanol": "6",
            "Pressure[hPa]": "6",
            "PM1.0": "6",
            "PM2.5": "6",
            "NC0.5": "6",
            "NC1.0": "6",
            "NC2.5": "6",
            "CNT": "6",
            "Fire Alarm": "2",
        },
        {
            "Unnamed: 0": "7",
            "UTC": "7",
            "Temperature[C]": "7",
            "Humidity[%]": "7",
            "TVOC[ppb]": "7",
            "eCO2[ppm]": "7",
            "Raw H2": "7",
            "Raw Ethanol": "7",
            "Pressure[hPa]": "7",
            "PM1.0": "7",
            "PM2.5": "7",
            "NC0.5": "7",
            "NC1.0": "7",
            "NC2.5": "7",
            "CNT": "7",
            "Fire Alarm": "1",
        },
    ]
    df = pd.DataFrame(data * 10)
    # Add row with all outliers
    df = df.append(
        {
            "Unnamed: 0": "1",
            "UTC": "1",
            "Temperature[C]": "15",
            "Humidity[%]": "15",
            "TVOC[ppb]": "15",
            "eCO2[ppm]": "15",
            "Raw H2": "15",
            "Raw Ethanol": "15",
            "Pressure[hPa]": "15",
            "PM1.0": "1",
            "PM2.5": "1",
            "NC0.5": "1",
            "NC1.0": "1",
            "NC2.5": "1",
            "CNT": "1",
            "Fire Alarm": "1",
        },
        ignore_index=True,
    )
    # Add row with only half outliers
    df = df.append(
        {
            "Unnamed: 0": "100",
            "UTC": "100",
            "Temperature[C]": "100",
            "Humidity[%]": "100",
            "TVOC[ppb]": "100",
            "eCO2[ppm]": "100",
            "Raw H2": "100",
            "Raw Ethanol": "100",
            "Pressure[hPa]": "100",
            "PM1.0": "100",
            "PM2.5": "100",
            "NC0.5": "100",
            "NC1.0": "100",
            "NC2.5": "100",
            "CNT": "100",
            "Fire Alarm": "100",
        },
        ignore_index=True,
    )
    df = df.astype(int)
    df["Fire Alarm"] = df["Fire Alarm"] - 1

    return df


@pytest.fixture()
def Smoke_Config():
    Smoke_Config = SmokeConfig(
        model=ModelConfig("", "", False, False, None, False, 1, 0.5, True, 1, 1, {}),
        dataset=DatasetConfig("", "", "raw", "preprocess", "preprocess_outlines"),
        secret=None,
        path=PathConfig("", "", "", "", "", "", "", ""),
        test=None,
        predict=None,
    )
    return Smoke_Config


def test_get_outliers(df):
    df_outliers = data.get_outliers(df)
    assert isinstance(df_outliers, pd.Series)
    assert df_outliers.shape == (df.shape[0],)


@pytest.mark.parametrize(
    "outliers_numb",
    [
        (-1),  # uncorrect  test
        (0),  # uncorrect test
    ],
)
def test_cleaning_fail(df, Smoke_Config, outliers_numb):
    Smoke_Config.model.outliers_numb = outliers_numb
    with pytest.raises(AssertionError) as excinfo:
        cleaned_df, cleaned_df_outlines = data.cleaning(df, Smoke_Config)
    assert "value must be grater then 0" in str(excinfo.value)


@pytest.mark.parametrize(
    "outliers_numb",
    [
        (1),  # correct without outliers
        (6),  # correct without outliers
        (7),  # correct without one outliers
        (14),  # correct with outliers
    ],
)
def test_cleaning(df, Smoke_Config, outliers_numb):
    Smoke_Config.model.outliers_numb = outliers_numb
    with tempfile.TemporaryDirectory() as dp:
        base_path = Path(dp)
        Smoke_Config.path.data = str(base_path)
        cleaned_df, cleaned_df_outlines = data.cleaning(df, Smoke_Config)

        assert cleaned_df.shape == (df.shape[0], (df.shape[1] - 2))
        assert cleaned_df_outlines.shape[1] == (df.shape[1] - 2)
        if outliers_numb > 0 and outliers_numb < cleaned_df.shape[1]:
            assert cleaned_df_outlines.shape[0] != df.shape[0]
            if outliers_numb < int((df.shape[1] - 2) / 2):
                assert cleaned_df_outlines.shape[0] == (df.shape[0] - 2)
            else:
                assert cleaned_df_outlines.shape[0] == (df.shape[0] - 1)
        else:
            assert cleaned_df_outlines.shape[0] == df.shape[0]

        assert not {"unnamed_0", "utc"}.issubset(cleaned_df.columns)
        assert not {"unnamed_0", "utc"}.issubset(cleaned_df_outlines.columns)

        cleaned_df_read = pd.read_csv(base_path / Smoke_Config.dataset.preprocess)
        assert cleaned_df.eq(cleaned_df_read).all().all()
        cleaned_df_outlines_read = pd.read_csv(
            base_path / Smoke_Config.dataset.preprocess_without_outlines
        )
        assert cleaned_df_outlines.eq(cleaned_df_outlines_read).all().all()


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T, np.array([0, 0, 1, 1])),  # banalced
        (np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T, np.array([0, 0, 0, 1])),  # banalced
        (np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T, np.array([0, 1, 1, 1])),  # banalced
    ],
)
def test_oversample(X, y):
    X_resampled, y_resampled = data.oversample(X, y)
    assert X_resampled.shape[0] == y_resampled.shape[0]
    _, counts = np.unique(y_resampled, return_counts=True)
    counts_unique = np.unique(counts)
    assert counts_unique.shape[0] == 1


@pytest.mark.parametrize(
    "oversample",
    [
        (True),  # banalced
        (False),  # banalced
    ],
)
def test_get_data_splits(df, Smoke_Config, oversample):
    Smoke_Config.model.oversample = oversample
    with tempfile.TemporaryDirectory() as dp:
        base_path = Path(dp)
        Smoke_Config.path.data = str(base_path)

        _, cleaned_df_outlines = data.cleaning(df, Smoke_Config)
        X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
            cleaned_df_outlines.drop(columns=["fire_alarm"], axis=1),
            cleaned_df_outlines["fire_alarm"],
            use_oversample=Smoke_Config.model.oversample,
        )
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
        assert len(X_train) / float(len(cleaned_df_outlines)) == pytest.approx(
            0.7, abs=0.11
        )  # 0.7 ± 0.11
        assert len(X_val) / float(len(cleaned_df_outlines)) == pytest.approx(
            0.15, abs=0.05
        )  # 0.15 ± 0.05
        assert len(X_test) / float(len(cleaned_df_outlines)) == pytest.approx(
            0.15, abs=0.05
        )  # 0.15 ± 0.05


@pytest.mark.parametrize(
    "cleaned_exist, cleaned_outliers_exist",
    [
        (True, True),  # check if cleaned data exists
        (False, True),  # check if one of cleaned data don't exists
        (True, False),  # check if one of cleaned data don't exists
        (False, False),  # check if both cleaned data don't exists
    ],
)
def test_preprocess(df, Smoke_Config, cleaned_exist, cleaned_outliers_exist):
    with tempfile.TemporaryDirectory() as dp:
        base_path = Path(dp)
        Smoke_Config.path.data = str(base_path)

        _, _ = data.cleaning(df, Smoke_Config)
        if not cleaned_exist:
            (base_path / Smoke_Config.dataset.preprocess).unlink()
            assert not (base_path / Smoke_Config.dataset.preprocess).is_file()
        if not cleaned_outliers_exist:
            (base_path / Smoke_Config.dataset.preprocess_without_outlines).unlink()
            assert not (base_path / Smoke_Config.dataset.preprocess_without_outlines).is_file()

        X_train, X_val, X_test, y_train, y_val, y_test = data.preprocess(df, Smoke_Config)
        assert (base_path / Smoke_Config.dataset.preprocess).is_file()
        assert (base_path / Smoke_Config.dataset.preprocess_without_outlines).is_file()
