from __future__ import annotations
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import time
import os

# --- QVNM Proxy & Vector Sessions ---
from fastapi import UploadFile, File
import aiofiles, tempfile, numpy as np, uuid, httpx

JULIA_BASE = os.environ.get("JULIA_BASE", "http://localhost:9000")

# in-memory session store: {sid: {"V": np.ndarray (d,N), "ids": List[str]}}
QSESS: dict[str, dict] = {}

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Operator-Dianne Bridge", version="0.1.0")
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

@app.get("/", response_class=HTMLResponse)
def root():
    return ('<meta http-equiv="refresh" content="0; url=/ui/">'
            '<a href="/ui/">Open UI</a>')

# -----------------------------
# QVNM: upload vectors (JSONL or NPZ)
# -----------------------------
@app.post("/qvnm/upload_vectors")
async def qvnm_upload_vectors(file: UploadFile = File(...)):
    fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(file.filename or "")[1])
    os.close(fd)
    try:
        async with aiofiles.open(tmp, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                await f.write(chunk)

        ids, V = [], None

        if tmp.endswith(".jsonl"):
            vecs = []
            async with aiofiles.open(tmp, "r", encoding="utf-8") as f:
                async for line in f:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    ids.append(rec.get("id") or f"id{len(ids)}")
                    vecs.append(np.asarray(rec["vector"], dtype=np.float32))
            M = np.stack(vecs, axis=0)           # (N, d)
            M = M / (np.linalg.norm(M, axis=1, keepdims=True)+1e-12)
            V = M.T.astype(np.float32)          # (d, N)

        elif tmp.endswith(".npz"):
            dat = np.load(tmp, allow_pickle=False)
            ids = list(map(str, dat["ids"]))
            M = dat["vectors"].astype(np.float32)   # (N, d)
            M = M / (np.linalg.norm(M, axis=1, keepdims=True)+1e-12)
            V = M.T                                 # (d, N)
        else:
            return JSONResponse({"error":"Unsupported file type. Use .jsonl or .npz"}, status_code=400)

        sid = uuid.uuid4().hex[:12]
        QSESS[sid] = {"V": V, "ids": ids}
        return {"session": sid, "N": int(V.shape[1]), "d": int(V.shape[0])}
    finally:
        try: os.remove(tmp)
        except Exception: pass

# -----------------------------
# QVNM: estimate intrinsic dimension / entropy (proxy to Julia)
# -----------------------------
@app.post("/qvnm/estimate_id")
async def qvnm_estimate_id(session: str, mode: str = "local", k: int = 10,
                           gamma: float = 0.5, alpha: float = 0.5, boots: int = 8, r: int = 64):
    sess = QSESS.get(session)
    if not sess: return JSONResponse({"error":"bad session"}, status_code=400)
    V = sess["V"]
    payload = {
        "V": V.ravel(order="F").tolist(),
        "d": int(V.shape[0]), "N": int(V.shape[1]),
        "mode": mode, "k": k, "gamma": gamma, "alpha": alpha, "boots": boots, "r": r
    }
    async with httpx.AsyncClient(timeout=180.0) as cx:
        resp = await cx.post(f"{JULIA_BASE}/qvnm/estimate_id", json=payload)
        resp.raise_for_status()
        data = resp.json()
    sess["m_hat"] = data["m_hat"] if mode == "local" else [data["m_hat"]]*V.shape[1]
    sess["H_hat"] = data["H_hat"] if mode == "local" else [data["H_hat"]]*V.shape[1]
    return {"mode": data["mode"], "m_hat": sess["m_hat"], "H_hat": sess["H_hat"], "diag": data.get("diag")}

# -----------------------------
# helper: kNN on cosine
# -----------------------------
def _knn_cosine(V: np.ndarray, k: int = 10):
    S = (V.T @ V).clip(-1, 1)
    D = 1.0 - S
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, kth=k, axis=1)[:, :k]
    rows = np.arange(D.shape[0])[:, None]
    dsel = D[rows, idx]
    order = np.argsort(dsel, axis=1)
    idx_sorted = idx[rows, order]
    d_sorted = dsel[rows, order]
    return d_sorted.astype(np.float32).tolist(), idx_sorted.astype(int).tolist()

# -----------------------------
# QVNM: build + preview (proxy to Julia /qvnm/preview)
# -----------------------------
@app.post("/qvnm/build_preview")
async def qvnm_build_preview(session: str, k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3,
                             r: int = 2, k_eval: int = 10, bins: int = 20):
    sess = QSESS.get(session)
    if not sess: return JSONResponse({"error":"bad session"}, status_code=400)
    V = sess["V"]; N = V.shape[1]
    m_hat = sess.get("m_hat"); H_hat = sess.get("H_hat")
    if m_hat is None or H_hat is None:
        return JSONResponse({"error":"run estimate_id first"}, status_code=400)
    weights, neighbors = _knn_cosine(V, k=k)

    build_payload = {
        "V": V.ravel(order="F").tolist(), "d": int(V.shape[0]), "N": int(N),
        "neighbors": neighbors, "weights": weights,
        "m_hat": m_hat, "H_hat": H_hat,
        "lambda_m": lambda_m, "lambda_h": lambda_h,
        "r": r, "k_eval": k_eval, "bins": bins, "mode": "build"
    }
    async with httpx.AsyncClient(timeout=300.0) as cx:
        resp = await cx.post(f"{JULIA_BASE}/qvnm/preview", json=build_payload)
        resp.raise_for_status()
        prev = resp.json()

    sess["preview"] = prev
    return prev