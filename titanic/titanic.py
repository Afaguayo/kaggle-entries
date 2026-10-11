import sys, pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
d, out = sys.argv[1], sys.argv[2]
tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
df = pd.concat([tr, te], ignore_index=True)
df["Title"] = df.Name.str.extract(r",\s*([^\.]+)\.")[0].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
df.loc[~df.Title.isin(["Mr","Mrs","Miss","Master"]), "Title"] = "Rare"
df["Fam"] = df.SibSp + df.Parch + 1
df["Sex"] = (df.Sex == "male").astype(int)
df["Fare"] = df.Fare.fillna(df.Fare.median())
df["Age"] = df.Age.fillna(df.groupby("Title").Age.transform("median"))
df["Deck"] = df.Cabin.str[0].fillna("U")
df["TktN"] = df.groupby("Ticket").Ticket.transform("count")
df["Surname"] = df.Name.str.split(",").str[0]
# family/ticket-group survival feature (WCG-style), computed leave-one-out on train labels
df["GrpSurv"] = 0.5
for key in ["Surname", "Ticket"]:
    for _, g in df.groupby(key):
        if len(g) < 2: continue
        for i in g.index:
            others = g.drop(i).Survived.dropna()
            if len(others): df.loc[i, "GrpSurv"] = others.max() if others.max()==1 else (0 if df.loc[i,"GrpSurv"]==0.5 else df.loc[i,"GrpSurv"])
X = pd.get_dummies(df[["Pclass","Sex","Age","Fare","Fam","TktN","GrpSurv","Title","Embarked","Deck"]].fillna({"Embarked":"S"}), dtype=float)
n = len(tr); y = tr.Survived
models = {"hgb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200),
          "rf": RandomForestClassifier(500, max_depth=6, random_state=0),
          "lr": LogisticRegression(max_iter=2000, C=0.5)}
P = []
for k, m in models.items():
    print(k, cross_val_score(m, X[:n], y, cv=10).mean().round(4))
    P.append(m.fit(X[:n], y).predict_proba(X[n:])[:, 1])
pd.DataFrame({"PassengerId": te.PassengerId, "Survived": (np.mean(P, 0) > 0.5).astype(int)}).to_csv(out, index=False)
