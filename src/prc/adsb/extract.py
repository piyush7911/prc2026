"""Reduce one adsb.lol daily archive (globe_history, ODbL 1.0) to the positions inside a +-0.15 degree box around each airport in APTS:
one row per position (icao, reg, type, callsign, airport, t, lat, lon, on_ground, alt, gs) -> data/adsb_lol/<day>.parquet.
Parts already fetched by prc.adsb.download are parsed in place; missing parts are downloaded (preferred release tag, then replicas).
Usage: python -m prc.adsb.extract --day YYYY-MM-DD [--keep-raw] [--workers N]"""
import argparse, gzip, json, pathlib, shutil, subprocess, tarfile, time, io
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
ROOT = pathlib.Path("."); RAW = ROOT / "data" / "adsb_lol_raw"; OUT = ROOT / "data" / "adsb_lol"
BASE = "https://github.com/adsblol/globe_history_{yr}/releases/download/{tag}/{tag}.tar.{part}"
APTS = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619), "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785),
        "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
PAD = 0.15; LAT = np.array([v[0] for v in APTS.values()]); LON = np.array([v[1] for v in APTS.values()]); NAMES = np.array(list(APTS))
LATMIN, LATMAX, LONMIN, LONMAX = LAT.min() - PAD, LAT.max() + PAD, LON.min() - PAD, LON.max() + PAD


def parse_member(raw: bytes):
    if raw[:2] == b"\x1f\x8b": raw = gzip.decompress(raw)
    try: d = json.loads(raw)
    except Exception: return []
    base = d.get("timestamp"); tr = d.get("trace")
    if base is None or not tr: return []
    n = len(tr); lat = np.array([p[1] if len(p) > 2 and p[1] is not None else np.nan for p in tr], dtype=float)
    lon = np.array([p[2] if len(p) > 2 and p[2] is not None else np.nan for p in tr], dtype=float)
    cand = (lat >= LATMIN) & (lat <= LATMAX) & (lon >= LONMIN) & (lon <= LONMAX)
    if not cand.any(): return []
    idx = np.where(cand)[0]; hit = np.full(len(idx), -1)
    for k, (la, lo) in enumerate(zip(LAT, LON)):
        m = (np.abs(lat[idx] - la) <= PAD) & (np.abs(lon[idx] - lo) <= PAD); hit[m & (hit < 0)] = k
    idx = idx[hit >= 0]; hit = hit[hit >= 0]
    if len(idx) == 0: return []
    icao, reg, typ = d.get("icao"), d.get("r"), d.get("t"); out = []; cs = None; last_cs_i = -1
    # callsign: carried in p[8] dict when it changes; propagate forward
    cs_at = {}
    for i, p in enumerate(tr):
        if len(p) > 8 and isinstance(p[8], dict) and p[8].get("flight"): cs = p[8]["flight"].strip(); cs_at[i] = cs
    cur = None; keys = sorted(cs_at); ki = 0
    for i, k in zip(idx, hit):
        while ki < len(keys) and keys[ki] <= i: cur = cs_at[keys[ki]]; ki += 1
        p = tr[i]; alt = p[3]; ground = 1 if (alt == "ground" or alt is None) else 0
        altf = float(alt) if isinstance(alt, (int, float)) else np.nan
        if not ground and (np.isnan(altf) or altf > 3000): continue
        out.append((icao, reg, typ, cur, NAMES[k], base + p[0], lat[i], lon[i], ground, altf, float(p[4]) if isinstance(p[4], (int, float)) else np.nan))
    return out


def parse_chunk(blobs): 
    rows = []
    for b in blobs: rows.extend(parse_member(b))
    return rows





def remote_size(url):
    """GitHub release assets ignore HEAD after the 302, but answer a 1-byte range with Content-Range."""
    r = subprocess.run(["curl", "-sL", "--max-time", "90", "-r", "0-0", "-D", "-", "-o", "/dev/null", url],
                       capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if ln.lower().startswith("content-range:") and "/" in ln:
            try: return int(ln.split("/")[-1].strip())
            except ValueError: pass
    return 0


def size_ok(url, dst):
    n = remote_size(url)
    return True if n == 0 else dst.stat().st_size == n


def fetch_parallel(url, dst, nchunk=8):
    """GitHub throttles a single stream (observed 20 KB/s after heavy use) but serves HTTP 206 ranges.
    Pull the asset as N parallel ranges and concatenate; fall back to one stream if ranges are refused."""
    size = remote_size(url)
    if size <= 0:
        r = subprocess.run(["curl", "-sfL", "--retry", "8", "--retry-delay", "10", "--retry-all-errors",
                            "--speed-time", "60", "--speed-limit", "300000", "-o", str(dst), url])
        return r.returncode == 0
    tmp = dst.parent / (dst.name + ".parts"); tmp.mkdir(exist_ok=True)
    step = size // nchunk + 1
    procs = []
    for i in range(nchunk):
        lo = i * step; hi = min(size - 1, (i + 1) * step - 1)
        if lo > hi: break
        cp = tmp / f"{i:03d}"
        if cp.exists() and cp.stat().st_size == (hi - lo + 1): continue
        procs.append(subprocess.Popen(["curl", "-sfL", "-r", f"{lo}-{hi}", "--retry", "10", "--retry-delay", "5",
                                       "--retry-all-errors", "--speed-time", "45", "--speed-limit", "150000",
                                       "-o", str(cp), url]))
    ok = all(p.wait() == 0 for p in procs)
    parts = sorted(tmp.glob("[0-9]*"))
    got = sum(p.stat().st_size for p in parts)
    if not ok or got != size:
        print(f"[dl] parallel fetch incomplete ({got}/{size}); retrying next attempt", flush=True)
        return False
    with open(dst, "wb") as o:
        for p_ in parts:
            with open(p_, "rb") as fsrc: shutil.copyfileobj(fsrc, o, 1 << 22)
    for p_ in parts: p_.unlink()
    tmp.rmdir()
    return dst.stat().st_size == size


PREF = ROOT / "data"
REPLICAS = ("prod-0", "staging-0", "prod-0tmp", "staging-0tmp")


def preferred_urls(day):
    """adsblol/globe_history_{yr}/PREFERRED_RELEASES.txt names the replica to use per day (one line of comma-separated part URLs)."""
    key = f"/v{day.replace('-', '.')}-planes-"
    for yr in (day[:4], str(int(day[:4]) + 1)):   # year-boundary days (e.g. 2025-12-31) are published in the NEXT year's repo
        fn = PREF / f"adsb_lol_preferred_{yr}.txt"
        if not fn.exists():
            subprocess.run(["curl", "-sL", "--max-time", "60", "-o", str(fn), f"https://raw.githubusercontent.com/adsblol/globe_history_{yr}/main/PREFERRED_RELEASES.txt"])
        for ln in open(fn) if fn.exists() else []:
            if key in ln: return [u.strip() for u in ln.strip().split(",") if u.strip()]
    return []


def replica_urls(day, suffix):
    """Part URLs for one replica tag: split parts (.tar.aa, .ab, ...) if present, else the unsplit .tar, else []."""
    tag = f"v{day.replace('-', '.')}-planes-readsb-{suffix}"
    for yr in (day[:4], str(int(day[:4]) + 1)):
        base = f"https://github.com/adsblol/globe_history_{yr}/releases/download/{tag}/{tag}.tar"
        if remote_size(base + ".aa") > 0:
            out = []
            for part in ("aa", "ab", "ac", "ad", "ae"):
                if remote_size(f"{base}.{part}") > 0: out.append(f"{base}.{part}")
                else: break
            return out
        if remote_size(base) > 0: return [base]
    return []


def candidates(day):
    """Preferred replica first (PREFERRED_RELEASES.txt), then every other replica tag of the same day, de-duplicated."""
    seen, out = set(), []
    pref = preferred_urls(day)
    if pref: out.append(pref); seen.add(pref[0].split("/")[-2])
    for suf in REPLICAS:
        tag = f"v{day.replace('-', '.')}-planes-readsb-{suf}"
        if tag in seen: continue
        u = replica_urls(day, suf)
        if u: out.append(u); seen.add(tag)
    return out


def parse_paths(day, paths, workers, t0):
    cat = subprocess.Popen(["cat", *map(str, paths)], stdout=subprocess.PIPE); rows = []; n = 0; batch = []; futures = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        try:
            with tarfile.open(fileobj=cat.stdout, mode="r|") as tf:
                for m in tf:
                    if not m.isfile() or "trace" not in m.name: continue
                    f = tf.extractfile(m)
                    if f is None: continue
                    batch.append(f.read()); n += 1
                    if len(batch) >= 2000:
                        futures.append(ex.submit(parse_chunk, batch)); batch = []
                        if len(futures) >= workers * 2:
                            rows.extend(futures.pop(0).result())
                    if n % 50000 == 0: print(f"[{day}] {n:,} traces read ({time.time() - t0:.0f}s)", flush=True)
        finally:
            cat.stdout.close(); cat.wait()
        if batch: futures.append(ex.submit(parse_chunk, batch))
        for fu in futures: rows.extend(fu.result())
    return rows, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--day", required=True); ap.add_argument("--keep-raw", action="store_true"); ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    cands = candidates(a.day)
    if not cands: raise SystemExit(f"[{a.day}] no replica found (preferred list + {REPLICAS})")
    print(f"[{a.day}] replicas: " + " | ".join(c[0].split("/")[-2] for c in cands), flush=True)
    for urls in cands:
        tag = urls[0].split("/")[-2]; paths = []; ok = True
        for url in urls:
            dst = RAW / url.split("/")[-1]
            if dst.exists() and dst.stat().st_size == 0: dst.unlink()
            if dst.exists() and not size_ok(url, dst): dst.unlink()
            if not dst.exists() and not fetch_parallel(url, dst): ok = False; print(f"[{a.day}] download failed: {url}", flush=True); break
            paths.append(dst); print(f"[{a.day}] have {dst.name} {dst.stat().st_size / 2**30:.2f} GiB ({time.time() - t0:.0f}s)", flush=True)
        if not ok: continue
        try:
            rows, n = parse_paths(a.day, paths, a.workers, t0)
        except Exception as e:   # corrupt replica (zlib / truncated tar): the other replica of the same day is usually intact
            print(f"[{a.day}] replica {tag} unreadable ({type(e).__name__}: {e}); trying next replica", flush=True)
            for p in paths: p.unlink(missing_ok=True)
            continue
        df = pd.DataFrame(rows, columns=["icao", "reg", "type", "callsign", "airport", "t", "lat", "lon", "on_ground", "alt", "gs"]); df["t"] = pd.to_datetime(df.t, unit="s")
        df.to_parquet(OUT / f"{a.day}.parquet", index=False)
        print(f"[{a.day}] replica {tag}: parsed {n:,} traces -> {len(df):,} in-box samples, {df.icao.nunique():,} aircraft; by airport {df.groupby('airport').icao.nunique().to_dict()} ({time.time() - t0:.0f}s)", flush=True)
        if not a.keep_raw:
            for p in paths: p.unlink(missing_ok=True)
        return
    raise SystemExit(f"[{a.day}] all replicas failed")


if __name__ == "__main__":
    main()
