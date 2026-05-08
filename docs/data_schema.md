# Data Schema

## train.csv

| Column | Type | Description |
| --- | --- | --- |
| `영업일자` | date string | Sales date. Parsed with `pandas.to_datetime`. |
| `영업장명_메뉴명` | string | Combined venue and menu identifier. |
| `매출수량` | numeric | Daily sales quantity target. |

## test/TEST_XX.csv

Same columns as `train.csv`. Each file should contain the observed period used to predict the following 7 days.

## sample_submission.csv

The first column is `영업일자`, formatted like `TEST_00+1일`. Remaining columns should match menu names from training data.

## Engineered Features

The pipeline creates the following features from `영업일자`.

| Feature | Description |
| --- | --- |
| `weekday` | Monday=0 through Sunday=6. |
| `month` | Calendar month. |
| `week` | ISO calendar week. |
| `is_weekend` | 1 for Saturday/Sunday, else 0. |
| `is_holiday` | Korean holiday flag from `holidayskr` if installed, otherwise `holidays.KR`. |

