#!/usr/bin/env python3
import argparse
import json
import urllib.request
from typing import Any, Dict


def post_json(url: str, obj: Dict[str, Any], timeout: int = 25) -> Dict[str, Any]:
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--julia-url", default="http://localhost:8099")
    ap.add_argument("--adj")
    args = ap.parse_args()

    with open(args.report, "r", encoding="utf-8") as f:
        ccl = json.load(f)
    out = post_json(args.julia_url + "/coherence", ccl)
    print("\n== Coherence ==\n", json.dumps(out, indent=2))

    if args.adj:
        with open(args.adj, "r", encoding="utf-8") as f:
            adj = json.load(f)
        opt = post_json(args.julia_url + "/optimize", adj)
        print("\n== Optimize ==\n", json.dumps(opt, indent=2))


if __name__ == "__main__":
    main()