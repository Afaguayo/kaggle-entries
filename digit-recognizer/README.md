# Digit Recognizer (MNIST)

**Competition:** [https://www.kaggle.com/competitions/digit-recognizer](https://www.kaggle.com/competitions/digit-recognizer)

| | |
|---|---|
| Public leaderboard score | **0.99632** |
| Metric | Accuracy (higher is better) |
| Local validation | Holdout accuracy 0.9962 |

## Approach
PyTorch CNN with random affine augmentation, 3 seeds averaged, trained on Apple MPS.

## Reproduce
```bash
kaggle competitions download -c digit-recognizer -p data/digit-recognizer && unzip data/digit-recognizer/*.zip -d data/digit-recognizer
python digits.py data/digit-recognizer submission.csv
../submit.sh digit-recognizer submission.csv "my message"
```
