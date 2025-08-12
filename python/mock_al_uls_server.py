"""
Minimal FastAPI mock for al-ULS /optimize so you can test the Julia bridge end-to-end.
Run:
  uvicorn python.mock_al_uls_server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()


class OptimizeIn(BaseModel):
    adjacency: List[List[float]]
    labels: Optional[List[str]] = None
    mode: Optional[str] = "kfp"
    beta: Optional[float] = 0.8


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/optimize")
def optimize(inp: OptimizeIn):
    adj = inp.adjacency
    n = len(adj)
    out_sums = [(i, sum(adj[i])) for i in range(n)]
    routing = [i for i, _ in sorted(out_sums, key=lambda t: t[1], reverse=True)]
    return {
        "routing": routing,
        "objective": sum(sum(row) for row in adj),
        "notes": f"mock-al-ULS mode={inp.mode} beta={inp.beta}",
    }