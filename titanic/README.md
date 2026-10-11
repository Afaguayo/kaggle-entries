# Titanic - Machine Learning from Disaster

**Competition:** [https://www.kaggle.com/competitions/titanic](https://www.kaggle.com/competitions/titanic)

| | |
|---|---|
| Public leaderboard score | **0.78229** |
| Metric | Accuracy (higher is better) |
| Local validation | 10-fold CV accuracy 0.849 |

## Approach
Soft vote of HistGradientBoosting + RandomForest + LogisticRegression on title, family-size and ticket-group survival features.

## Reproduce
```bash
kaggle competitions download -c titanic -p data/titanic && unzip data/titanic/*.zip -d data/titanic
python titanic.py data/titanic submission.csv
../submit.sh titanic submission.csv "my message"
```
