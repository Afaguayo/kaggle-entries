# Disaster tweets (F1). TF-IDF word+char + logistic regression, threshold tuned on OOF. args: data_dir out_csv
import sys, re, pandas as pd, numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score
d, out = sys.argv[1], sys.argv[2]
tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
def clean(df): return (df.keyword.fillna("").str.replace("%20", " ") + " | " + df.text.str.replace(r"https?://\S+", " URL ", regex=True)).str.lower()
a, b = clean(tr), clean(te); y = tr.target.values
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True); c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
X = hstack([w.fit_transform(a), c.fit_transform(a)]).tocsr(); Xt = hstack([w.transform(b), c.transform(b)]).tocsr()
m = LogisticRegression(C=3, max_iter=3000)
oof = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
ths = np.arange(.3, .65, .01); f = [f1_score(y, oof > t) for t in ths]; t = ths[int(np.argmax(f))]
print("OOF F1", round(max(f), 4), "thr", round(t, 2))
pd.DataFrame({"id": te.id, "target": (m.fit(X, y).predict_proba(Xt)[:, 1] > t).astype(int)}).to_csv(out, index=False)
