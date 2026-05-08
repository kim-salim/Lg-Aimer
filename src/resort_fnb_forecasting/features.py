# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd


DATE_COL = "영업일자"
MENU_COL = "영업장명_메뉴명"
TARGET_COL = "매출수량"

FEATURE_COLUMNS = [
    "weekday",
    "month",
    "week",
    "is_weekend",
    "is_holiday",
    TARGET_COL,
]

REQUIRED_COLUMNS = [DATE_COL, MENU_COL, TARGET_COL]


def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> None:
    required = required or REQUIRED_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _holiday_flags(dates: pd.Series) -> list[int]:
    date_values = list(dates)

    try:
        from holidayskr import is_holiday
    except ImportError:
        is_holiday = None

    if is_holiday is not None:
        return [int(is_holiday(d.strftime("%Y-%m-%d"))) for d in date_values]

    try:
        import holidays
    except ImportError as exc:
        raise ImportError(
            "Install either holidayskr or holidays to create Korean holiday features."
        ) from exc

    years = sorted({d.year for d in date_values})
    kr_holidays = holidays.KR(years=years)
    return [int(d in kr_holidays) for d in date_values]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekday, month, ISO week, weekend, and Korean holiday features."""
    validate_columns(df)

    out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(out[DATE_COL]):
        out[DATE_COL] = pd.to_datetime(out[DATE_COL])

    out["weekday"] = out[DATE_COL].dt.weekday
    out["month"] = out[DATE_COL].dt.month
    out["week"] = out[DATE_COL].dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["is_holiday"] = _holiday_flags(out[DATE_COL].dt.date)
    return out

