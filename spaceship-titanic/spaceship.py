# Spaceship Titanic (binary accuracy). args: data_dir out_csv
import sys, pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
d, out = sys.argv[1], sys.argv[2]
tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
df = pd.concat([tr, te], ignore_index=True); n = len(tr); y = tr.Transported.astype(int).values
sp = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df["Group"] = df.PassengerId.str[:4]; df["GSize"] = df.groupby("Group").Group.transform("count")
cab = df.Cabin.str.split("/", expand=True); df["Deck"], df["CNum"], df["Side"] = cab[0], pd.to_numeric(cab[1]), cab[2]
# CryoSleep passengers spend nothing; fill missing spend accordingly
for c in sp: df.loc[df[c].isna() & (df.CryoSleep == True), c] = 0
df.loc[df.CryoSleep.isna() & (df[sp].sum(1) > 0), "CryoSleep"] = False
df["Spend"] = df[sp].sum(1); df["NoSpend"] = (df.Spend == 0).astype(int)
for c in sp + ["Spend"]: df[c] = np.log1p(df[c])
df["Surname"] = df.Name.str.split().str[-1]; df["FamSize"] = df.groupby("Surname").Surname.transform("count")
df["CNumBin"] = (df.CNum // 300)
cats = ["HomePlanet", "Destination", "Deck", "Side"]
X = df[["CryoSleep", "VIP", "Age", "GSize", "CNum", "CNumBin", "NoSpend", "FamSize"] + sp + ["Spend"] + cats].copy()
X["CryoSleep"] = X.CryoSleep.astype(float); X["VIP"] = X.VIP.astype(float)
for c in cats: X[c] = X[c].astype("category")
P, acc = np.zeros(len(X) - n), []
for seed in range(5):
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(X[:n], y)):
        m = HistGradientBoostingClassifier(learning_rate=.03, max_iter=400, max_leaf_nodes=20, min_samples_leaf=30, l2_regularization=1,
                                           categorical_features="from_dtype", random_state=seed).fit(X.iloc[a], y[a])
        acc.append((m.predict(X.iloc[b]) == y[b]).mean()); P += m.predict_proba(X[n:])[:, 1] / 25
print("CV acc", np.mean(acc).round(4))
pd.DataFrame({"PassengerId": te.PassengerId, "Transported": P > .5}).to_csv(out, index=False)
