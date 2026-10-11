# Playground Series S6E10 - Predicting Airline Satisfaction

**Competition:** [https://www.kaggle.com/competitions/playground-series-s6e10](https://www.kaggle.com/competitions/playground-series-s6e10)

| | |
|---|---|
| Public leaderboard score | **0.95806** |
| Metric | ROC AUC (higher is better) |
| Local validation | 5-fold OOF ROC AUC 0.9584 |

## Approach
HistGradientBoosting with early stopping, 5-fold average, plus mean service rating and total delay features.

## Reproduce
```bash
kaggle competitions download -c playground-series-s6e10 -p data/playground-series-s6e10 && unzip data/playground-series-s6e10/*.zip -d data/playground-series-s6e10
python playground.py data/playground-series-s6e10 submission.csv
../submit.sh playground-series-s6e10 submission.csv "my message"
```
