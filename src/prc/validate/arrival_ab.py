"""A/B test of the same-runway arrival taxi-in columns (prc.validate.extra_ideas.arrival_rwy_features) given to the model itself.
Same fast LightGBM (255 leaves, lr 0.1, 400 rounds, fixed) on the six 2-month folds of the pipeline, with and without the 14 `ra_*` columns.
Gate: lower RMSE in each of Jan, Feb, Jun and Jul 2025 and on the four-month total, gain not carried by fewer than 100 rows.
Usage: python -m prc.validate.arrival_ab"""
import numpy as np, pandas as pd, lightgbm as lgb
from prc.models import pipeline as P
def main():
    f, fb, _ = P.load(); f = f.reset_index(drop=True); A = pd.read_parquet(f"{P.OUT}/rwy_arr_feats.parquet")
    f = f.merge(A, on="MVT_ID_mvt", how="left"); ra = [c for c in A.columns if c.startswith("ra_")]
    lab = f[~f.is_rank & (f.y >= 0)].reset_index(drop=True); cats = [c for c in P.CATS if c in fb]; b = P.base_zf(lab); t = lab.y.to_numpy() - b
    prm = dict(P.HP, learning_rate=0.1, num_leaves=255, min_data_in_leaf=100); res = {}
    for name, feats in (("without", fb), ("with", fb + ra)):
        p = np.zeros(len(lab))
        for fm in P.FOLDS:
            te = lab.month.isin(fm).to_numpy(); tr = ~te
            p[te] = lgb.train(prm, lgb.Dataset(lab.loc[tr, feats], t[tr], categorical_feature=cats), 400).predict(lab.loc[te, feats]) + b[te]
            P.log(f"{name} fold {fm}: {P.rmse((lab.y.to_numpy() - p)[te]):.2f}")
        res[name] = p
    y = lab.y.to_numpy(); ok = P.good(lab); mo = lab.month.to_numpy(); r = P.rmse
    for m in (1, 2, 6, 7, "four", "all"):
        s = ok & (np.isin(mo, (1, 2, 6, 7)) if m == "four" else (np.ones(len(lab), bool) if m == "all" else mo == m))
        print(f"month {m}: without {r((y - res['without'])[s]):.3f}  with {r((y - res['with'])[s]):.3f}  diff {r((y - res['with'])[s]) - r((y - res['without'])[s]):+.3f}")
    g = (y - res["without"]) ** 2 - (y - res["with"]) ** 2; s = ok & np.isin(mo, (1, 2, 6, 7)); gg = np.sort(g[s])[::-1]
    print(f"top-100 rows carry {gg[:100].sum() / g[s].sum():.0%} of the four-month gain")
if __name__ == "__main__":
    main()
