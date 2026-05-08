# Resort F&B Sales Forecasting

LG Aimers 리조트 식음업장 결제 데이터를 활용해 메뉴별 향후 7일 매출수량을 예측하는 LSTM 기반 시계열 예측 프로젝트입니다.

입력은 최근 28일의 메뉴별 매출 및 캘린더 feature이며, 출력은 이후 7일의 매출수량입니다. 기본 모델은 메뉴 embedding을 포함한 Seq2Seq LSTM이고, 성능 개선을 위해 seed bagging 방식의 5-model ensemble을 사용해 보았습니다.

## Project Structure

```text
.
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── data/
│   ├── .gitkeep
│   └── test/
│       └── .gitkeep
├── docs/
│   ├── data_schema.md
│   └── experiment_notes.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── submissions/
│   └── .gitkeep
└── src/
    └── resort_fnb_forecasting/
        ├── __init__.py
        ├── cli.py
        ├── config.py
        ├── dataset.py
        ├── features.py
        ├── model.py
        ├── predict.py
        ├── train.py
        └── utils.py
```

## Data Placement

대회 데이터는 GitHub에 올리지 않는 것을 권장합니다. 아래 위치에 직접 배치한 뒤 실행하세요.

```text
data/
├── train.csv
├── sample_submission.csv
└── test/
    ├── TEST_00.csv
    ├── TEST_01.csv
    └── ...
```

필수 컬럼은 다음과 같습니다.

- `영업일자`
- `영업장명_메뉴명`
- `매출수량`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

개발 모드로 설치하려면 다음을 사용할 수 있습니다.

```bash
pip install -e .
```

## Train And Predict

기본 설정은 `epoch=40`, `ensemble_size=5`입니다.

```bash
python main.py --mode all
```

학습만 실행:

```bash
python main.py --mode train
```

이미 학습된 모델로 예측만 실행:

```bash
python main.py --mode predict
```

30 epoch 단일/앙상블 실험처럼 설정을 바꿔 실행할 수 있습니다.

```bash
python main.py --mode all --epochs 30 --ensemble-size 5
```

## Model

사용 feature:

- 메뉴 ID embedding
- weekday
- month
- ISO week
- is_weekend
- is_holiday
- 매출수량

구조:

- Encoder: LSTM(`menu embedding + calendar/sales features`)
- Decoder: autoregressive LSTM
- Output: 7-day sales forecast

## Experiment Notes

기록된 실험 점수:

| Version | Setting | Score |
| --- | --- | ---: |
| Single Seq2Seq | 50 epoch | 0.7814005475 |
| Single Seq2Seq | 30 epoch | 0.765854557 |
| Ensemble Seq2Seq | 40 epoch, 5 models, no weighting | 0.6432123097 |

현재 정리된 기본값은 가장 좋은 기록이 있었던 ensemble 설정을 기준으로 합니다.

## Outputs

학습 결과:

```text
models/
├── best_seq2seq_seed42.pth
├── best_seq2seq_seed43.pth
├── ...
└── ensemble_manifest.json
```

예측 결과:

```text
submissions/submission_ensemble_5models_YYYYMMDD_HHMMSS.csv
```

