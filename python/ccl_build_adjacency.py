#!/usr/bin/env python3
"""
Build an adjacency matrix JSON from a CCL report.
Usage:
  python python/ccl_build_adjacency.py report.json --out adjacency.json
"""
import argparse
import json
from typing import Any, Dict, List


def build_adjacency(report: Dict[str, Any]) -> Dict[str, Any]:
    fqs: List[str] = [f["qualname"] for f in report.get("functions", []) if f.get("arity", 0) == 1]
    idx = {q: i for i, q in enumerate(fqs)}
    n = len(fqs)
    adj: List[List[float]] = [[0.0] * n for _ in range(n)]
    for p in report.get("pipelines", []):
        pair = p.get("pair", [None, None])
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        qf, qg = pair
        if qf in idx and qg in idx:
            i, j = idx[qf], idx[qg]
            try:
                agr = float(p.get("agreement_rate", 0.0))
            except Exception:
                agr = 0.0
            w = max(0.0, 1.0 - agr)
            adj[i][j] = max(adj[i][j], w)
            adj[j][i] = max(adj[j][i], w)
    return {"adjacency": adj, "labels": fqs, "mode": "kfp", "beta": 0.8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--out", default="adjacency.json")
    args = ap.parse_args()
    with open(args.report, "r", encoding="utf-8") as f:
        rep = json.load(f)
    out = build_adjacency(rep)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} with {len(out['labels'])} nodes.")


if __name__ == "__main__":
    main()