# Store Sales - Time Series Forecasting

**Competition:** [https://www.kaggle.com/competitions/store-sales-time-series-forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

| | |
|---|---|
| Public leaderboard score | **0.42371** |
| Metric | RMSLE (lower is better) |
| Local validation | Last-16-days RMSLE 0.394 (14-day-mean baseline 0.536) |

## Approach
One global HistGradientBoosting model predicting all 16 days ahead directly from rolling means, same-weekday means, promotions and holidays, on log1p(sales).

## Reproduce
```bash
kaggle competitions download -c store-sales-time-series-forecasting -p data/store-sales-time-series-forecasting && unzip data/store-sales-time-series-forecasting/*.zip -d data/store-sales-time-series-forecasting
python store_sales.py data/store-sales-time-series-forecasting submission.csv
../submit.sh store-sales-time-series-forecasting submission.csv "my message"
```
