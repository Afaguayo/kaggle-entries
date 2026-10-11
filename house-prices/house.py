# House-price regression (Ames). args: data_dir out_csv metric(mae|rmsle)
import sys, pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_predict
d, out = sys.argv[1], sys.argv[2]
tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
tr = tr[~((tr.GrLivArea > 4000) & (tr.SalePrice < 300000))]
y = np.log1p(tr.SalePrice.values); n = len(tr)
df = pd.concat([tr.drop(columns="SalePrice"), te], ignore_index=True)
df["TotalSF"] = df.TotalBsmtSF.fillna(0) + df["1stFlrSF"] + df["2ndFlrSF"]
df["Baths"] = df.FullBath + .5*df.HalfBath + df.BsmtFullBath.fillna(0) + .5*df.BsmtHalfBath.fillna(0)
df["Age"] = df.YrSold - df.YearBuilt; df["RemAge"] = df.YrSold - df.YearRemodAdd
df["QualSF"] = df.OverallQual * df.GrLivArea
num = df.select_dtypes("number").columns.drop("Id")
X = pd.get_dummies(df.drop(columns="Id"), dtype=float)
Xn = X.copy(); Xn[num] = Xn[num].fillna(Xn[num].median())
sk = Xn[num].skew().abs(); Xn[sk[sk > .75].index] = np.log1p(Xn[sk[sk > .75].index].clip(lower=0))
hgb = HistGradientBoostingRegressor(learning_rate=.03, max_iter=1500, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=1)
rdg = make_pipeline(RobustScaler(), RidgeCV(alphas=np.logspace(-1, 2.5, 30)))
kf = KFold(5, shuffle=True, random_state=0)
ph = cross_val_predict(hgb, X[:n], y, cv=kf); pr = cross_val_predict(rdg, Xn[:n], y, cv=kf)
for nm, p in [("hgb", ph), ("ridge", pr), ("blend", .5*ph+.5*pr)]:
    print(nm, "rmse(log)", np.sqrt(((p-y)**2).mean()).round(4), "MAE$", np.abs(np.expm1(p)-np.expm1(y)).mean().round(0))
p = .5*hgb.fit(X[:n], y).predict(X[n:]) + .5*rdg.fit(Xn[:n], y).predict(Xn[n:])
pd.DataFrame({"Id": te.Id, "SalePrice": np.expm1(p)}).to_csv(out, index=False)
