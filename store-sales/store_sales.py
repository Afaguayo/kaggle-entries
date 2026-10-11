# Direct multi-horizon global HGB on log1p(sales). args: data_dir out_csv
import sys, pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
d, out = sys.argv[1], sys.argv[2]
tr = pd.read_csv(f"{d}/train.csv", parse_dates=["date"]); te = pd.read_csv(f"{d}/test.csv", parse_dates=["date"])
tr = tr[tr.date >= "2016-01-01"]
full = pd.date_range("2016-01-01", "2017-08-31")
S = np.log1p(tr.pivot_table(index=["store_nbr","family"], columns="date", values="sales")).reindex(columns=full)
for dd in full[(full.month == 12) & (full.day == 25)]: S[dd] = S[dd - pd.Timedelta(days=7)]  # stores closed on Christmas
S = S.fillna(0)
PR = pd.concat([tr, te]).pivot_table(index=["store_nbr","family"], columns="date", values="onpromotion").reindex(index=S.index, columns=full).fillna(0).values
hol = pd.read_csv(f"{d}/holidays_events.csv", parse_dates=["date"])
natl = set(hol[(hol.locale == "National") & (~hol.transferred) & (hol.type != "Work Day")].date)
stores = pd.read_csv(f"{d}/stores.csv").set_index("store_nbr")
idx = S.index.to_frame(index=False); st = idx.store_nbr.values
static = {"fam": idx.family.astype("category").cat.codes.values, "store": st,
          "stype": stores.loc[st, "type"].astype("category").cat.codes.values, "clus": stores.loc[st, "cluster"].values}
V = S.values; col = {c: i for i, c in enumerate(full)}
def feats(origin):
    o = col[origin] + 1
    base = {f"m{w}": V[:, o-w:o].mean(1) for w in (3, 7, 14, 28, 56, 112)}
    base.update(zero28=(V[:, o-28:o] == 0).mean(1), std28=V[:, o-28:o].std(1), promo28=PR[:, o-28:o].mean(1), **static)
    rows = []
    for h in range(1, 17):
        t = origin + pd.Timedelta(days=h); ti = col[t]; n = len(V)
        f = dict(base, h=np.full(n, h), dow=np.full(n, t.dayofweek), dom=np.full(n, t.day), hol=np.full(n, int(t in natl)),
                 promo=np.log1p(PR[:, ti]), y=V[:, ti] if t <= pd.Timestamp("2017-08-15") else np.nan)
        same_dow = [ti - 7*j for j in range(1, 10) if ti - 7*j < o][:8]  # same weekday, inside history
        f["dowm4"] = V[:, same_dow[:4]].mean(1); f["dowm8"] = V[:, same_dow].mean(1)
        rows.append(pd.DataFrame(f))
    return pd.concat(rows, ignore_index=True)
def origins(last): return [last - pd.Timedelta(days=7*k) for k in range(1, 22)] + [last - pd.Timedelta(days=364 + 7*k) for k in range(-2, 3)]
cat = list(static)
def fit(df, seed=0):
    df = df[df.y.notna()]
    F = [c for c in df.columns if c != "y"]
    return HistGradientBoostingRegressor(learning_rate=.05, max_iter=500, max_leaf_nodes=63, min_samples_leaf=50, random_state=seed,
                                         categorical_features=[c in ("fam", "stype") for c in F]).fit(df[F], df.y), F
def pred(mF, df):
    m, F = mF; p = np.clip(m.predict(df[F]), 0, None); p[df.zero28.values == 1] = 0; return p
val_o = pd.Timestamp("2017-07-30")
mF = fit(pd.concat([feats(o) for o in origins(val_o)])); va = feats(val_o); p = pred(mF, va)
print("val RMSLE", np.sqrt(((p - va.y)**2).mean()).round(4), "| naive m14:", np.sqrt(((va.m14 - va.y)**2).mean()).round(4), flush=True)
last = pd.Timestamp("2017-08-15")
trn = pd.concat([feats(o) for o in origins(last)]); tf = feats(last)
p = np.mean([pred(fit(trn, s), tf) for s in range(3)], 0)
tf["date"] = [last + pd.Timedelta(days=int(h)) for h in tf.h]; tf["store_nbr"] = np.tile(st, 16); tf["family"] = np.tile(idx.family.values, 16)
tf["sales"] = np.expm1(p)
sub = te.merge(tf[["date", "store_nbr", "family", "sales"]], on=["date", "store_nbr", "family"], how="left")
print("rows", len(sub), "missing", sub.sales.isna().sum()); sub[["id", "sales"]].fillna(0).to_csv(out, index=False)
