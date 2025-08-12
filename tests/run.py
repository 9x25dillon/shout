import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PY = ROOT / "python"


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_examples_adjacency_json_valid():
    p = EXAMPLES / "adjacency.json"
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert_true("adjacency" in data and isinstance(data["adjacency"], list), "missing adjacency list")
    assert_true("labels" in data and isinstance(data["labels"], list), "missing labels list")
    n = len(data["adjacency"]) or 0
    assert_true(all(len(row) == n for row in data["adjacency"]), "adjacency is not square")


def test_examples_have_no_json_comments():
    for p in EXAMPLES.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                s = line.lstrip()
                assert_true(not (s.startswith("#") or s.startswith("//")), f"comment in JSON {p}:{i}")


def test_adjacency_builder():
    in_path = EXAMPLES / "report.min.json"
    out_path = EXAMPLES / "adjacency.test.out.json"
    cmd = [sys.executable, str(PY / "ccl_build_adjacency.py"), str(in_path), "--out", str(out_path)]
    subprocess.check_call(cmd)
    with out_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert_true("adjacency" in data, "missing adjacency")
    assert_true("labels" in data, "missing labels")
    n = len(data["labels"])
    A = data["adjacency"]
    assert_true(len(A) == n and all(len(row) == n for row in A), "adj size mismatch")
    # For the supplied report, there are 2 unary functions; expect a 2x2 matrix with zeros on diagonal
    assert_true(n == 2, f"expected 2 labels, got {n}")
    assert_true(A[0][0] == 0.0 and A[1][1] == 0.0, "expected zero diagonal")


if __name__ == "__main__":
    tests = [
        test_examples_adjacency_json_valid,
        test_examples_have_no_json_comments,
        test_adjacency_builder,
    ]
    for t in tests:
        t()
        print(f"OK: {t.__name__}")
    print("All tests passed.")