"""Gate for a change that requires re-training the whole pipeline (e.g. a new ADS-B extraction): compares two complete output folders.
Reports, on the 2025 out-of-fold predictions, the change of RMSE in each month of GATE_MONTHS and over them (excluding the five corrupt records),
the concentration statistic G100 (share of the net gain carried by the 100 most-improved records) over all records and over the records
whose input column CHANGED_COL differs between the two folders, and the size of the change on the scored rows against its size on 2025.
For such a change the other records differ only by re-training noise, so G100 is taken over the records whose inputs changed.
Usage (repository root): python -m prc.validate.compare <reference output folder> <new output folder> [changed column, default ads_as_leave]"""
import importlib, os, sys
import numpy as np, pandas as pd

def run(out):
    os.environ["PRC_OUT"] = out; os.environ["PRC_FEAT"] = f"{out}/features_ads.parquet"
    import prc.models.pipeline as P
    P = importlib.reload(P); P._MT = P._SH = P._UC = None
    f, _, _ = P.load(); lab = f[~f.is_rank].reset_index(drop=True); rk = f[f.is_rank].reset_index(drop=True)
    q, _ = P._assemble(lab, "oof"); r, _ = P._assemble(rk, "pred")
    return P, pd.DataFrame({"id": lab.MVT_ID_mvt, "y": lab.y, "m": lab.month, "ok": np.asarray(P.good(lab)), "q": q}), pd.DataFrame({"id": rk.MVT_ID_mvt, "r": r})

if __name__ == "__main__":
    ref, new = sys.argv[1], sys.argv[2]; col = sys.argv[3] if len(sys.argv) > 3 else "ads_as_leave"
    P, a, ar = run(ref); _, b, br = run(new)
    A = a.merge(b[["id", "q"]], on="id", suffixes=("", "_n")); B = ar.merge(br, on="id", suffixes=("", "_n"))
    rm = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))
    for m in P.GATE_MONTHS:
        s = A.ok & (A.m == m); print(f"month {m}: {rm((A.y - A.q)[s]):.2f} -> {rm((A.y - A.q_n)[s]):.2f}")
    s4 = A.ok & A.m.isin(P.GATE_MONTHS); g = ((A.y - A.q) ** 2 - (A.y - A.q_n) ** 2)
    print(f"gate months together: {rm((A.y - A.q)[s4]):.2f} -> {rm((A.y - A.q_n)[s4]):.2f}")
    c0 = pd.read_parquet(f"{ref}/adsb_feats.parquet", columns=["MVT_ID_mvt", col]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")[col].reindex(A.id)
    c1 = pd.read_parquet(f"{new}/adsb_feats.parquet", columns=["MVT_ID_mvt", col]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")[col].reindex(A.id)
    chg = ~((c0.to_numpy() == c1.to_numpy()) | (c0.isna().to_numpy() & c1.isna().to_numpy()))
    for nm, s in (("all records", s4), (f"records whose {col} changed", s4 & chg)):
        gg = np.sort(g[s].to_numpy())[::-1]; tot = gg.sum()
        print(f"G100 over {nm} ({int(s.sum()):,}): {gg[:100].sum() / tot:.2f} (net gain {tot:.3g} s^2)" if tot > 0 else f"{nm}: no net gain")
    d25 = rm((A.q_n - A.q)[A.m.isin(P.GATE_MONTHS)]); d26 = rm(B.r_n - B.r)
    print(f"size of the change: 2025 {d25:.2f} s, scored rows {d26:.2f} s, ratio {d26 / d25:.2f}; mean shift on the scored rows {float((B.r_n - B.r).mean()):+.2f} s")
