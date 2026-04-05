from __future__ import annotations

import re
from typing import Dict

import pandas as pd

FEATURE_ORDER = [
    "temperature_2_m_above_gnd",
    "relative_humidity_2_m_above_gnd",
    "mean_sea_level_pressure_MSL",
    "total_precipitation_sfc",
    "snowfall_amount_sfc",
    "total_cloud_cover_sfc",
    "high_cloud_cover_high_cld_lay",
    "medium_cloud_cover_mid_cld_lay",
    "low_cloud_cover_low_cld_lay",
    "shortwave_radiation_backwards_sfc",
    "wind_speed_10_m_above_gnd",
    "wind_direction_10_m_above_gnd",
    "wind_speed_80_m_above_gnd",
    "wind_direction_80_m_above_gnd",
    "wind_speed_900_mb",
    "wind_direction_900_mb",
    "wind_gust_10_m_above_gnd",
    "angle_of_incidence",
    "zenith",
    "azimuth",
]

FEATURE_DEFAULTS = {
    "temperature_2_m_above_gnd": 15.17,
    "relative_humidity_2_m_above_gnd": 51.14,
    "mean_sea_level_pressure_MSL": 1019.32,
    "total_precipitation_sfc": 0.03,
    "snowfall_amount_sfc": 0.001,
    "total_cloud_cover_sfc": 33.96,
    "high_cloud_cover_high_cld_lay": 14.85,
    "medium_cloud_cover_mid_cld_lay": 20.39,
    "low_cloud_cover_low_cld_lay": 20.70,
    "shortwave_radiation_backwards_sfc": 388.02,
    "wind_speed_10_m_above_gnd": 16.11,
    "wind_direction_10_m_above_gnd": 195.64,
    "wind_speed_80_m_above_gnd": 18.87,
    "wind_direction_80_m_above_gnd": 192.60,
    "wind_speed_900_mb": 16.25,
    "wind_direction_900_mb": 193.86,
    "wind_gust_10_m_above_gnd": 20.47,
    "angle_of_incidence": 50.89,
    "zenith": 59.93,
    "azimuth": 168.63,
}

COLUMN_ALIASES = {
    "temp": "temperature_2_m_above_gnd",
    "temperature": "temperature_2_m_above_gnd",
    "humidity": "relative_humidity_2_m_above_gnd",
    "relative_humidity": "relative_humidity_2_m_above_gnd",
    "cloud_cover": "total_cloud_cover_sfc",
    "cloudiness": "total_cloud_cover_sfc",
    "radiation": "shortwave_radiation_backwards_sfc",
    "irradiance": "shortwave_radiation_backwards_sfc",
    "solar_irradiance": "shortwave_radiation_backwards_sfc",
    "wind_speed": "wind_speed_10_m_above_gnd",
    "windspeed": "wind_speed_10_m_above_gnd",
    "pressure": "mean_sea_level_pressure_MSL",
    "msl_pressure": "mean_sea_level_pressure_MSL",
    "precipitation": "total_precipitation_sfc",
    "snowfall": "snowfall_amount_sfc",
    "wind_direction": "wind_direction_10_m_above_gnd",
    "wind_gust": "wind_gust_10_m_above_gnd",
    "aoi": "angle_of_incidence",
}

TARGET_COL = "generated_power_kw"


if set(FEATURE_ORDER) != set(FEATURE_DEFAULTS):
    missing = set(FEATURE_ORDER) - set(FEATURE_DEFAULTS)
    extras = set(FEATURE_DEFAULTS) - set(FEATURE_ORDER)
    raise ValueError(
        f"FEATURE_DEFAULTS mismatch. Missing: {sorted(missing)} | Extra: {sorted(extras)}"
    )


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _build_alias_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}

    for feature in FEATURE_ORDER:
        lookup[_normalize(feature)] = feature

    for alias, feature in COLUMN_ALIASES.items():
        lookup[_normalize(alias)] = feature

    return lookup


ALIAS_LOOKUP = _build_alias_lookup()


def align_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aligns arbitrary input columns to the exact model feature schema.

    - Renames known aliases (case-insensitive)
    - Drops target column if present
    - Fills missing values/features using FEATURE_DEFAULTS
    - Returns columns in FEATURE_ORDER exactly
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("align_features expects a pandas DataFrame")

    aligned = df.copy()

    # Drop target using case-insensitive matching.
    drop_cols = [col for col in aligned.columns if _normalize(col) == _normalize(TARGET_COL)]
    if drop_cols:
        aligned = aligned.drop(columns=drop_cols)

    rename_map = {}
    for col in aligned.columns:
        normalized = _normalize(col)
        canonical = ALIAS_LOOKUP.get(normalized)
        if canonical:
            rename_map[col] = canonical

    if rename_map:
        aligned = aligned.rename(columns=rename_map)

    # If multiple aliases map to the same canonical feature, merge by first non-null.
    if aligned.columns.duplicated().any():
        deduped = {}
        for col_name in dict.fromkeys(aligned.columns):
            same_name_cols = aligned.loc[:, aligned.columns == col_name]
            deduped[col_name] = same_name_cols.bfill(axis=1).iloc[:, 0]
        aligned = pd.DataFrame(deduped, index=aligned.index)

    for feature in FEATURE_ORDER:
        default_value = FEATURE_DEFAULTS[feature]

        if feature not in aligned.columns:
            aligned[feature] = default_value
        else:
            aligned[feature] = pd.to_numeric(aligned[feature], errors="coerce").fillna(
                default_value
            )

    return aligned[FEATURE_ORDER]
