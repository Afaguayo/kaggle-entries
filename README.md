# Kaggle entries

My solo submissions to Kaggle competitions — one folder per competition with the full training script, the approach and the score it got.

Kaggle profile: [kaggle.com/afaguayo](https://www.kaggle.com/afaguayo)

| Competition | Metric | Public score | Local validation | Code |
|---|---|---|---|---|
| [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) | Accuracy | **0.78229** | 10-fold CV accuracy 0.849 | [`titanic/`](titanic/) |
| [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | RMSE of log price | **0.12172** | 5-fold CV RMSLE 0.110 | [`house-prices/`](house-prices/) |
| [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course) | Mean absolute error in $ | **12,704.96** | 5-fold CV MAE ≈ $13.0k | [`housing-prices-learn/`](housing-prices-learn/) |
| [Digit Recognizer (MNIST)](https://www.kaggle.com/competitions/digit-recognizer) | Accuracy | **0.99632** | Holdout accuracy 0.9962 | [`digit-recognizer/`](digit-recognizer/) |
| [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) | Accuracy | **0.80547** | 5x5-fold CV accuracy 0.813 | [`spaceship-titanic/`](spaceship-titanic/) |
| [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) | F1 | **0.80049** | 5-fold OOF F1 0.770 | [`nlp-disaster-tweets/`](nlp-disaster-tweets/) |
| [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | RMSLE | **0.42371** | Last-16-days RMSLE 0.394 (14-day-mean baseline 0.536) | [`store-sales/`](store-sales/) |
| [Playground Series S6E10 - Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10) | ROC AUC | **0.95806** | 5-fold OOF ROC AUC 0.9584 | [`airline-satisfaction-s6e10/`](airline-satisfaction-s6e10/) |

## How it's set up
- Each script takes `data_dir out_csv` and writes a ready-to-upload submission file.
- Every model is validated locally (cross-validation or a time-based holdout) before submitting, so the local number is an honest estimate rather than a leaderboard-tuned one.
- `submit.sh <competition> <csv> "<message>"` uploads through the Kaggle REST API, reading the token from `~/.kaggle/access_token` (never committed).
- Competition data isn't included (Kaggle's rules forbid redistributing it); download it with the Kaggle CLI.

## Stack
Python 3.9 · pandas · scikit-learn (HistGradientBoosting, linear models) · PyTorch (CNN on Apple MPS)
