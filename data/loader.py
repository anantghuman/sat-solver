from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

ROOT = Path(__file__).resolve().parents[1] / "data" / "satlib"


def iter_instances(bucket: str, root: Path = ROOT) -> Iterator[tuple[Path, Optional[bool]]]:
    """Yield (cnf_path, is_sat) for every .cnf in `bucket`.

    is_sat is True for uf* buckets, False for uuf*, None if the naming
    convention isn't recognized."""
    bucket_dir = root / bucket
    if not bucket_dir.is_dir():
        raise FileNotFoundError(f"bucket not found: {bucket_dir}")
    label: Optional[bool]
    if bucket.startswith("uuf"):
        label = False
    elif bucket.startswith("uf"):
        label = True
    else:
        label = None
    for p in sorted(bucket_dir.glob("*.cnf")):
        yield p, label
