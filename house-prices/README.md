# House Prices - Advanced Regression Techniques

**Competition:** [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

| | |
|---|---|
| Public leaderboard score | **0.12172** |
| Metric | RMSE of log price (lower is better) |
| Local validation | 5-fold CV RMSLE 0.110 |

## Approach
Blend of HistGradientBoosting and RidgeCV on log(SalePrice), outliers removed, engineered area/age/quality features.

## Reproduce
```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/house-prices-advanced-regression-techniques && unzip data/house-prices-advanced-regression-techniques/*.zip -d data/house-prices-advanced-regression-techniques
python house.py data/house-prices-advanced-regression-techniques submission.csv rmsle
../submit.sh house-prices-advanced-regression-techniques submission.csv "my message"
```
