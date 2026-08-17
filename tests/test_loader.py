from pathlib import Path

import pytest

from data.loader import iter_instances


def test_iter_instances_labels(tmp_path):
    for bucket, expected in [("uf20-91", True), ("uuf50-218", False)]:
        (tmp_path / bucket).mkdir()
        (tmp_path / bucket / "a.cnf").write_text("p cnf 1 1\n1 0\n")
        (tmp_path / bucket / "b.cnf").write_text("p cnf 1 1\n-1 0\n")
        results = list(iter_instances(bucket, root=tmp_path))
        assert len(results) == 2
        assert all(is_sat is expected for _, is_sat in results)


def test_iter_instances_missing_bucket(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_instances("does-not-exist", root=tmp_path))
