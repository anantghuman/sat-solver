"""Download and unpack SATLIB benchmark buckets under data/satlib/<bucket>/.

Buckets we care about (uniform random 3-SAT at various sizes):
  uf20-91, uf50-218, uuf50-218, uf75-325, uuf75-325.

Usage: python -m data.download_satlib [--bucket NAME ...]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

BASE = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT"

BUCKETS = {
    "uf20-91": f"{BASE}/uf20-91.tar.gz",
    "uf50-218": f"{BASE}/uf50-218.tar.gz",
    "uuf50-218": f"{BASE}/uuf50-218.tar.gz",
    "uf75-325": f"{BASE}/uf75-325.tar.gz",
    "uuf75-325": f"{BASE}/uuf75-325.tar.gz",
}

ROOT = Path(__file__).resolve().parents[1] / "data" / "satlib"


def already_populated(bucket_dir: Path) -> bool:
    return bucket_dir.is_dir() and any(bucket_dir.glob("*.cnf"))


def fetch(bucket: str, url: str) -> None:
    dest = ROOT / bucket
    if already_populated(dest):
        print(f"[skip] {bucket} already at {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {url}")
    with urlopen(url) as resp:
        data = resp.read()
    print(f"[unpack] {len(data)/1024:.1f} KiB -> {dest}")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isreg():
                continue
            name = os.path.basename(member.name)
            if not name.endswith(".cnf"):
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            (dest / name).write_bytes(src.read())
    n = sum(1 for _ in dest.glob("*.cnf"))
    print(f"[done]  {bucket}: {n} .cnf files")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", action="append", choices=list(BUCKETS.keys()))
    args = ap.parse_args(argv)
    targets = args.bucket or list(BUCKETS.keys())
    for b in targets:
        fetch(b, BUCKETS[b])
    return 0


if __name__ == "__main__":
    sys.exit(main())
