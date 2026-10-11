# Housing Prices Competition for Kaggle Learn Users

**Competition:** [https://www.kaggle.com/competitions/home-data-for-ml-course](https://www.kaggle.com/competitions/home-data-for-ml-course)

| | |
|---|---|
| Public leaderboard score | **12,704.96** |
| Metric | Mean absolute error in $ (lower is better) |
| Local validation | 5-fold CV MAE ≈ $13.0k |

## Approach
Same model as House Prices (same Ames data), evaluated on MAE.

## Reproduce
```bash
kaggle competitions download -c home-data-for-ml-course -p data/home-data-for-ml-course && unzip data/home-data-for-ml-course/*.zip -d data/home-data-for-ml-course
python house.py data/home-data-for-ml-course submission.csv mae
../submit.sh home-data-for-ml-course submission.csv "my message"
```
