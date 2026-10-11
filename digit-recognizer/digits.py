# MNIST CNN ensemble with light augmentation. args: data_dir out_csv
import sys, time, pandas as pd, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
d, out = sys.argv[1], sys.argv[2]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
X = torch.tensor(tr.drop(columns="label").values, dtype=torch.float32).view(-1,1,28,28)/255.
y = torch.tensor(tr.label.values); Xt = torch.tensor(te.values, dtype=torch.float32).view(-1,1,28,28)/255.
def net():
    c = lambda i,o: [nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU()]
    return nn.Sequential(*c(1,32),*c(32,32),nn.MaxPool2d(2),nn.Dropout(.25),*c(32,64),*c(64,64),nn.MaxPool2d(2),nn.Dropout(.25),
                         nn.Flatten(),nn.Linear(64*49,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.4),nn.Linear(256,10))
def aug(b):
    th = torch.zeros(len(b),2,3); a = (torch.rand(len(b))-.5)*0.35; s = 1+(torch.rand(len(b))-.5)*.2
    th[:,0,0]=s*torch.cos(a); th[:,0,1]=-s*torch.sin(a); th[:,1,0]=s*torch.sin(a); th[:,1,1]=s*torch.cos(a); th[:,:,2]=(torch.rand(len(b),2)-.5)*.2
    return F.grid_sample(b, F.affine_grid(th, b.shape, align_corners=False), align_corners=False)
def train(Xa, ya, ep, seed):
    torch.manual_seed(seed); m = net().to(dev); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    steps = ep*((len(Xa)+127)//128); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    for e in range(ep):
        m.train(); perm = torch.randperm(len(Xa))
        for i in range(0, len(Xa), 128):
            idx = perm[i:i+128]; xb = aug(Xa[idx]).to(dev); loss = F.cross_entropy(m(xb), ya[idx].to(dev), label_smoothing=.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()
@torch.no_grad()
def proba(m, Z): return torch.cat([F.softmax(m(Z[i:i+1000].to(dev)),1).cpu() for i in range(0,len(Z),1000)])
t=time.time(); g = torch.Generator().manual_seed(0); p = torch.randperm(len(X), generator=g); va, trn = p[:4000], p[4000:]
m = train(X[trn], y[trn], 15, 0); print("holdout acc", (proba(m, X[va]).argmax(1)==y[va]).float().mean().item(), f"{time.time()-t:.0f}s", flush=True)
P = proba(m, Xt)
for s in (1,2): P += proba(train(X, y, 15, s), Xt); print("seed", s, "done", flush=True)
pd.DataFrame({"ImageId": np.arange(1, len(Xt)+1), "Label": P.argmax(1).numpy()}).to_csv(out, index=False)
