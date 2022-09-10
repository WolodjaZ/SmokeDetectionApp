from pydantic import BaseModel


class SmokeFeatures(BaseModel):
    """Smoke features."""

    temperature_c: float
    humidity: float
    tvoc_ppb: int
    e_co_2_ppm: int
    raw_h_2: int
    raw_ethanol: int
    pressure_h_pa: float
    pm_1_0: float
    pm_2_5: float
    nc_0_5: float
    nc_1_0: float
    nc_2_5: float
    cnt: int

    class Config:
        schema_extra = {
            "example": {
                "temperature_c": 22.0,
                "humidity": 0.0,
                "tvoc_ppb": 0,
                "e_co_2_ppm": 0,
                "raw_h_2": 0,
                "raw_ethanol": 0,
                "pressure_h_pa": 0.0,
                "pm_1_0": 0.0,
                "pm_2_5": 0.0,
                "nc_0_5": 0.0,
                "nc_1_0": 0.0,
                "nc_2_5": 0.0,
                "cnt": 0,
            }
        }
