# Spaceship Titanic

**Competition:** [https://www.kaggle.com/competitions/spaceship-titanic](https://www.kaggle.com/competitions/spaceship-titanic)

| | |
|---|---|
| Public leaderboard score | **0.80547** |
| Metric | Accuracy (higher is better) |
| Local validation | 5x5-fold CV accuracy 0.813 |

## Approach
HistGradientBoosting, 5 seeds x 5 folds; cabin deck/side, travel-group, CryoSleep-aware spend imputation.

## Reproduce
```bash
kaggle competitions download -c spaceship-titanic -p data/spaceship-titanic && unzip data/spaceship-titanic/*.zip -d data/spaceship-titanic
python spaceship.py data/spaceship-titanic submission.csv
../submit.sh spaceship-titanic submission.csv "my message"
```
