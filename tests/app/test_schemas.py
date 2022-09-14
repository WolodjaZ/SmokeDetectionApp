import pytest

from app import schemas


def test_SmokeFeatures():
    smoke_features = schemas.SmokeFeatures(
        temperature_c=22.0,
        humidity=0.0,
        tvoc_ppb=0,
        e_co_2_ppm=0,
        raw_h_2=0,
        raw_ethanol=0,
        pressure_h_pa=0.0,
        pm_1_0=0.0,
        pm_2_5=0.0,
        nc_0_5=0.0,
        nc_1_0=0.0,
        nc_2_5=0.0,
        cnt=0,
    )
    assert smoke_features is not None


@pytest.mark.parametrize(
    "temp, hum, tvoc, e_co_2, raw_h_2, raw_ethanol, pressure_h_pa, pm_1_0, pm_2_5, nc_0_5, nc_1_0, nc_2_5, cnt",
    [
        (None, None, None, None, None, None, None, None, None, None, None, None, None),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None),
    ],
)
def test_fail_not_enough_data_SmokeFeatures(
    temp,
    hum,
    tvoc,
    e_co_2,
    raw_h_2,
    raw_ethanol,
    pressure_h_pa,
    pm_1_0,
    pm_2_5,
    nc_0_5,
    nc_1_0,
    nc_2_5,
    cnt,
):
    with pytest.raises(Exception):
        schemas.SmokeFeatures(
            temperature_c=temp,
            humidity=hum,
            tvoc_ppb=tvoc,
            e_co_2_ppm=e_co_2,
            raw_h_2=raw_h_2,
            raw_ethanol=raw_ethanol,
            pressure_h_pa=pressure_h_pa,
            pm_1_0=pm_1_0,
            pm_2_5=pm_2_5,
            nc_0_5=nc_0_5,
            nc_1_0=nc_1_0,
            nc_2_5=nc_2_5,
            cnt=cnt,
        )


def test_wrong_data_format_SmokeFeatures():
    input = schemas.SmokeFeatures(
        temperature_c=22.0,
        humidity=0.0,
        tvoc_ppb=0,
        e_co_2_ppm=0,
        raw_h_2=0,
        raw_ethanol=0,
        pressure_h_pa=0.0,
        pm_1_0=0,
        pm_2_5=0,
        nc_0_5=0,
        nc_1_0=0,
        nc_2_5=0,
        cnt=0.0,
    )

    assert input.nc_2_5 == 0.0
    assert input.cnt == 0


def test_fail_wrong_data_format_SmokeFeatures():
    with pytest.raises(Exception):
        schemas.SmokeFeatures(
            temperature_c=22.0,
            humidity=0.0,
            tvoc_ppb=0,
            e_co_2_ppm=0,
            raw_h_2=0,
            raw_ethanol=0,
            pressure_h_pa=0.0,
            pm_1_0=0,
            pm_2_5=0,
            nc_0_5=0,
            nc_1_0=0,
            nc_2_5=0,
            cnt="SSS",
        )
