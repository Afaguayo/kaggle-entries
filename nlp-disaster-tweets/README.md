# Natural Language Processing with Disaster Tweets

**Competition:** [https://www.kaggle.com/competitions/nlp-getting-started](https://www.kaggle.com/competitions/nlp-getting-started)

| | |
|---|---|
| Public leaderboard score | **0.80049** |
| Metric | F1 (higher is better) |
| Local validation | 5-fold OOF F1 0.770 |

## Approach
TF-IDF word (1-2) + char (2-5) n-grams into LogisticRegression, decision threshold tuned on out-of-fold F1.

## Reproduce
```bash
kaggle competitions download -c nlp-getting-started -p data/nlp-getting-started && unzip data/nlp-getting-started/*.zip -d data/nlp-getting-started
python nlp.py data/nlp-getting-started submission.csv
../submit.sh nlp-getting-started submission.csv "my message"
```
