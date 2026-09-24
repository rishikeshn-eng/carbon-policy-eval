"""Offline checks on the dataset. Run: python -m pytest tests/ (or python tests/test_dataset.py)."""
import json
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "ccts_eval.jsonl"
ROWS = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]


def test_schema():
    for r in ROWS:
        assert {"id", "type", "topic", "input", "target", "source"} <= r.keys(), r["id"]
        assert r["type"] in {"factual", "false_premise"}
        assert r["source"].startswith("https://")


def test_unique_ids():
    ids = [r["id"] for r in ROWS]
    assert len(ids) == len(set(ids))


def test_false_premise_targets_say_so():
    for r in ROWS:
        if r["type"] == "false_premise":
            assert r["target"].startswith("False premise"), r["id"]


def test_counts():
    kinds = [r["type"] for r in ROWS]
    assert kinds.count("factual") == 25 and kinds.count("false_premise") == 15


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"ok: {len(ROWS)} rows")
