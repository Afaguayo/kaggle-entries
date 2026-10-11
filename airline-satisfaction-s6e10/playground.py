# Airline satisfaction (binary, ROC AUC). args: data_dir out_csv
import sys, pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
d, out = sys.argv[1], sys.argv[2]
tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
y = tr.satisfaction.astype(str).str.lower().isin(["true", "1", "satisfied"]).astype(int).values
cats = ["Gender", "Customer Type", "Type of Travel", "Class"]
def prep(df):
    X = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    for c in cats: X[c] = X[c].astype("category")
    svc = [c for c in X.columns if X[c].dtype != "category" and X[c].max() <= 5 and c not in ("Age",)]
    X["svc_mean"] = X[svc].mean(1); X["delay_tot"] = X["Departure Delay in Minutes"] + X["Arrival Delay in Minutes"].fillna(0)
    return X
X, Xt = prep(tr), prep(te)
for c in cats: Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)
print("rows", len(X), "pos rate", y.mean().round(3), flush=True)
oof, pt = np.zeros(len(X)), np.zeros(len(Xt))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)):
    m = HistGradientBoostingClassifier(learning_rate=.05, max_iter=2000, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1,
                                       categorical_features="from_dtype", early_stopping=True, validation_fraction=.1, n_iter_no_change=100, random_state=k)
    m.fit(X.iloc[a], y[a]); oof[b] = m.predict_proba(X.iloc[b])[:, 1]; pt += m.predict_proba(Xt)[:, 1] / 5
    print("fold", k, roc_auc_score(y[b], oof[b]).round(5), "iters", m.n_iter_, flush=True)
print("OOF AUC", roc_auc_score(y, oof).round(5))
pd.DataFrame({"id": te.id, "satisfaction": pt}).to_csv(out, index=False)
