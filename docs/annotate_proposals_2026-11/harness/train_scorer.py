"""Learned auto-accept scorer: leave-one-12X-experiment-out. python train_scorer.py VARIANT [target ok|id_ok]"""
import glob
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

variant = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else "ok"
PREC = 0.99
d = pd.concat([pd.read_parquet(p) for p in glob.glob(f"feats_{variant}_*.parquet")], ignore_index=True)
easy = pd.concat([pd.read_parquet(p) for p in glob.glob("easy_now_*.parquet")])
d = d.merge(easy[["exp", "a", "easy"]], on=["exp", "a"], how="left")
d["is12"] = d.exp.str.contains("_12X_")
DROP = {"exp", "a", "b", "ok", "id_ok", "n_err", "frames_ok", "reasons", "easy", "is12"}
FEATS = [c for c in d.columns if c not in DROP]
X = d[FEATS].astype(float).to_numpy()
y = d[target].astype(int).to_numpy()


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)


def threshold(p, yy, prec=PREC):
    """lowest score threshold whose accepted set has precision >= prec (scores sorted high to low)"""
    order = np.argsort(-p)
    cum = np.cumsum(yy[order]) / np.arange(1, len(order) + 1)
    ok = np.where(cum >= prec)[0]
    # largest k with precision >= prec, requiring at least 20 accepted
    ok = ok[ok >= 19]
    return p[order[ok.max()]] if len(ok) else 1.01


rows = []
d["p"] = np.nan
for test in sorted(d.exp[d.is12].unique()):
    tr = (d.exp != test).to_numpy()
    te = (d.exp == test).to_numpy()
    # threshold from out-of-fold scores of the training experiments
    groups = d.exp[tr].to_numpy()
    oof = np.zeros(tr.sum())
    for a, b in GroupKFold(n_splits=min(5, len(set(groups)))).split(X[tr], y[tr], groups):
        m = model().fit(X[tr][a], y[tr][a])
        oof[b] = m.predict_proba(X[tr][b])[:, 1]
    thr = threshold(oof, y[tr])
    m = model().fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    d.loc[te, "p"] = p
    acc = p >= thr
    g = d[te]
    e = g.easy.fillna(False).to_numpy()
    rows.append(dict(exp=test, stretches=te.sum(), frames=g.length.sum(), base_ok=100 * y[te].mean(), thr=round(thr, 3),
                     learned_n=acc.sum(), learned_prec=100 * y[te][acc].mean() if acc.any() else np.nan,
                     learned_frames=g.length[acc].sum(),
                     easy_n=e.sum(), easy_prec=100 * y[te][e].mean() if e.any() else np.nan, easy_frames=g.length[e].sum(),
                     machine_anchor_n=(acc & ~g.anchor_kind.to_numpy()).sum(),
                     machine_anchor_prec=100 * y[te][acc & ~g.anchor_kind.to_numpy()].mean() if (acc & ~g.anchor_kind.to_numpy()).any() else np.nan))
r = pd.DataFrame(rows)
tot = r[["stretches", "frames", "learned_n", "learned_frames", "easy_n", "easy_frames"]].sum()
tot["learned_prec"] = (r.learned_prec * r.learned_n).sum() / r.learned_n.sum()
tot["easy_prec"] = (r.easy_prec * r.easy_n).sum() / r.easy_n.sum()
pd.set_option("display.width", 250)
print(variant, target)
print(r.round(1).to_string())
print(tot.round(1).to_string())
d.to_pickle(f"scored_{variant}_{target}.pkl")
m = model().fit(X, y)
from sklearn.inspection import permutation_importance
imp = permutation_importance(m, X, y, n_repeats=3, random_state=0, scoring="average_precision")
print(pd.Series(imp.importances_mean, FEATS).sort_values(ascending=False).head(12).round(4).to_string())
