"""Blend weight and gate of the TabM partner on UNCOVERED matched rows (2025 out-of-fold). pred = (1 - w) * pipeline + w * TabM.
Weight rule: pooled optimum over the twelve months; checked by leaving each fold out. Gate: lower RMSE on uncovered rows in each of
Jan, Feb, Jun and Jul 2025 and on the four-month total, gain not carried by fewer than 100 rows. Writes outputs/final/tabm_weight.json.
Usage: python -m prc.validate.tabm_blend"""
import json, numpy as np, pandas as pd
from prc.models import pipeline as P
def main():
    f, _, _ = P.load(); lab = f[~f.is_rank].reset_index(drop=True); ids = lab.MVT_ID_mvt.to_numpy()
    p, c = P._assemble(lab, "oof"); ln = c["lane"]; y = lab.y.to_numpy(); mo = lab.month.to_numpy(); ok = P.good(lab)
    t = pd.read_parquet(f"{P.OUT}/oof_tabm.parquet").set_index("MVT_ID_mvt").pred.reindex(ids).to_numpy()
    unc = (ln == "uncovered") & ok & np.isfinite(t); r = P.rmse; grid = np.round(np.arange(0, 0.81, 0.05), 2)
    sc = lambda w, s: r((y - ((1 - w) * p + w * t))[s])
    print(f"uncovered rows {unc.sum():,}; pipeline {r((y-p)[unc]):.3f}, TabM alone {r((y-t)[unc]):.3f}, rms disagreement {r((p-t)[unc]):.1f}")
    print("w      " + " ".join(f"{w:6.2f}" for w in grid))
    for nm, s in (("12 mo", unc), ("4 mo", unc & np.isin(mo, (1, 2, 6, 7))), ("Jan", unc & (mo == 1)), ("Feb", unc & (mo == 2)), ("Jun", unc & (mo == 6)), ("Jul", unc & (mo == 7))):
        print(f"{nm:6s} " + " ".join(f"{sc(w, s):6.2f}" for w in grid))
    wpool = float(grid[int(np.argmin([sc(w, unc) for w in grid]))]); lfo = []
    for fm in P.FOLDS:
        tr = unc & ~np.isin(mo, fm); lfo.append(float(grid[int(np.argmin([sc(w, tr) for w in grid]))]))
    print(f"pooled optimum w = {wpool}; leave-fold-out optima {lfo}")
    w = wpool; gate = True
    for nm, s in (("Jan", unc & (mo == 1)), ("Feb", unc & (mo == 2)), ("Jun", unc & (mo == 6)), ("Jul", unc & (mo == 7)), ("four months", unc & np.isin(mo, (1, 2, 6, 7)))):
        d = sc(w, s) - sc(0, s); gate &= d < 0; print(f"  w={w}: {nm:12s} uncovered RMSE {sc(0, s):.3f} -> {sc(w, s):.3f} ({d:+.3f})")
    g = (y - p) ** 2 - (y - ((1 - w) * p + w * t)) ** 2; s4 = unc & np.isin(mo, (1, 2, 6, 7)); gg = np.sort(g[s4])[::-1]
    top = float(gg[:100].sum() / g[s4].sum()); allm = ok & (ln != "") ; pm = np.where(ln == "uncovered", (1 - w) * p + w * np.nan_to_num(t, nan=0) if False else (1 - w) * p + w * t, p)
    pm = np.where(np.isfinite(pm), pm, p); s_all = ok & np.isin(mo, (1, 2, 6, 7))
    print(f"top-100 rows carry {top:.0%} of the four-month gain | whole-file four-month RMSE {r((y-p)[s_all]):.3f} -> {r((y-pm)[s_all]):.3f}")
    keep = bool(gate and top < 0.5); json.dump({"w": w if keep else 0.0, "gate_passed": keep, "pooled_optimum": wpool, "leave_fold_out": lfo}, open(f"{P.OUT}/tabm_weight.json", "w"))
    print("GATE", "PASSED" if keep else "FAILED", "-> weight", w if keep else 0.0)
if __name__ == "__main__":
    main()
