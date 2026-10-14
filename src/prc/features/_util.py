"""Shared helper of the feature steps: time-stamped log line."""
import time
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
