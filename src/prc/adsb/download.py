"""Resumable parallel downloader for adsb.lol daily archives (github.com/adsblol/globe_history_YYYY releases, ODbL 1.0).

Each part file (.tar.aa, .tar.ab, ...) is fetched as N byte ranges in parallel; every range RESUMES from the bytes already on disk
after a dropped connection (no speed-limit aborts, no restart from zero, no switch to another replica while bytes are pending).
Complete part files land in data/adsb_lol_raw/, where prc.adsb.extract finds them ("have ...") and only parses.
Usage: python -m prc.adsb.download 2026-02-01 2026-02-02 ...   (or a month: 2026-02)"""
import calendar, subprocess, sys, time, pathlib
from concurrent.futures import ThreadPoolExecutor
from prc.adsb.extract import preferred_urls, replica_urls, remote_size

RAW = pathlib.Path("data/adsb_lol_raw"); RAW.mkdir(parents=True, exist_ok=True)
NCHUNK = 24

def _range(url, lo, hi, cp):
    """Download bytes lo..hi into cp, resuming from cp's current size until complete."""
    need = hi - lo + 1
    for attempt in range(200):
        have = cp.stat().st_size if cp.exists() else 0
        if have >= need: return True
        with open(cp, "ab") as o:
            subprocess.run(["curl", "-sfL", "--connect-timeout", "20", "-r", f"{lo + have}-{hi}", url], stdout=o)
        time.sleep(1 if cp.exists() and cp.stat().st_size > have else 5)
    return False

def fetch(url):
    dst = RAW / url.split("/")[-1]; size = remote_size(url)
    if dst.exists() and dst.stat().st_size == size: return dst
    tmp = RAW / (dst.name + ".r"); tmp.mkdir(exist_ok=True); step = size // NCHUNK + 1
    jobs = [(url, i * step, min(size - 1, (i + 1) * step - 1), tmp / f"{i:03d}") for i in range(NCHUNK) if i * step < size]
    with ThreadPoolExecutor(NCHUNK) as ex: ok = all(ex.map(lambda j: _range(*j), jobs))
    if not ok: raise RuntimeError(f"incomplete {url}")
    with open(dst, "wb") as o:
        for j in jobs: o.write(j[3].read_bytes())
    assert dst.stat().st_size == size, (dst, size)
    for j in jobs: j[3].unlink()
    tmp.rmdir(); return dst

def days(args):
    for a in args:
        if len(a) == 7:
            y, m = map(int, a.split("-")); yield from (f"{a}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1))
        else: yield a

if __name__ == "__main__":
    for day in days(sys.argv[1:]):
        if pathlib.Path(f"data/adsb_lol/{day}.parquet").exists(): continue
        urls = preferred_urls(day) or replica_urls(day, "prod-0") or replica_urls(day, "staging-0")
        t0 = time.time(); tot = 0
        for u in urls: tot += fetch(u).stat().st_size
        print(f"[{day}] downloaded {tot / 1e9:.2f} GB in {time.time() - t0:.0f}s ({tot / 1e6 / max(time.time() - t0, 1):.1f} MB/s)", flush=True)
        (RAW / f"{day}.ready").write_text("\n".join(u.split("/")[-1] for u in urls))
