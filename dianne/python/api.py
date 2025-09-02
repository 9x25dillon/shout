#!/usr/bin/env python3

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse


from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import aiofiles
import json
import csv
import io
import time


> main
import numpy as np
import httpx
import uuid
from typing import List, Dict, Any

app = FastAPI()
app.mount(
    "/ui",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static")),
    name="ui",
)

JULIA_BASE = os.environ.get("JULIA_BASE", "http://localhost:9000")

# Session memory: sid -> { V (d×N), ids (N), d, N, neighbors, weights, m_hat, H_hat }
QSESS: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return '<meta http-equiv="refresh" content="0;url=/ui/qvnm.html" />'


# ---------- helpers ----------
def _ensure_float32_col_unit(V: np.ndarray) -> np.ndarray:
    # V can be d×N or N×d; we expect d×N
    if V.ndim != 2:
        raise ValueError("Vectors must be a 2D array")
    d0, d1 = V.shape
    # if looks like N×d (more rows than cols), assume transpose
    if d0 > d1:
        V = V.T
    V = V.astype(np.float32, copy=False)
    norms = np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
    V = V / norms
    return V


def _knn_graph_from_V(V: np.ndarray, k: int = 10) -> tuple[list[list[int]], list[list[float]]]:
    # V is d×N unit-norm
    d, N = V.shape
    # cosine sim
    S = (V.T @ V).astype(np.float64)
    # ensure numerical cleanliness
    np.clip(S, -1.0, 1.0, out=S)
    neighbors: list[list[int]] = []
    weights: list[list[float]] = []
    # distance on sphere: chordal distance sqrt(2-2cos)
    for i in range(N):
        sims = S[i]
        # exclude self by setting to -inf
        sims_i = sims.copy()
        sims_i[i] = -np.inf
        # top-k by similarity
        nn_idx = np.argpartition(-sims_i, kth=min(k, N - 1) - 1)[: min(k, N - 1)]
        # sort by similarity desc
        nn_idx = nn_idx[np.argsort(-sims_i[nn_idx])]
        # convert to 1-based for Julia
        nbs = (nn_idx + 1).tolist()
        # weights are distances (cost)
        dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sims_i[nn_idx])).tolist()
        neighbors.append(nbs)
        weights.append(dist)
    return neighbors, weights


async def _post_json(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# ---------- text helpers ----------
def _texts_for_ids(sess: Dict[str, Any], id_list: list[str]) -> list[dict[str, str]]:
    ids: list[str] = sess.get("ids", []) or []
    texts: list[str] = sess.get("texts", []) or []
    pos: dict[str, int] = {ids[i]: i for i in range(len(ids))}
    out: list[dict[str, str]] = []
    for _id in id_list:
        i = pos.get(_id)
        if i is not None and i < len(texts):
            out.append({"id": _id, "text": texts[i]})
    return out


# ---------- text helpers ----------
def _texts_for_ids(sess: Dict[str, Any], id_list: list[str]) -> list[dict[str, str]]:
    ids: list[str] = sess.get("ids", []) or []
    texts: list[str] = sess.get("texts", []) or []
    pos: dict[str, int] = {ids[i]: i for i in range(len(ids))}
    out: list[dict[str, str]] = []
    for _id in id_list:
        i = pos.get(_id)
        if i is not None and i < len(texts):
            out.append({"id": _id, "text": texts[i]})
    return out


# ---------- upload vectors (.jsonl or .npz) ----------
@app.post("/qvnm/upload_vectors")
async def qvnm_upload_vectors(file: UploadFile = File(...)) -> JSONResponse:
    fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(file.filename or "")[1])
    os.close(fd)
    try:
        async with aiofiles.open(tmp, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await f.write(chunk)
        ids: list[str] = []
        V: np.ndarray | None = None
        if tmp.endswith(".jsonl"):
            vecs: list[np.ndarray] = []
            texts: list[str] = []
            async with aiofiles.open(tmp, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ids.append(str(rec.get("id", f"id{len(ids)}")))
                    vecs.append(np.asarray(rec["vector"], dtype=np.float32))
                    texts.append(str(rec.get("text", "")))
            M = np.stack(vecs, axis=0)
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            V = M.T.astype(np.float32, copy=False)  # d×N


                    texts.append(str(rec.get("text", "")))
            M = np.stack(vecs, axis=0)
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            V = M.T.astype(np.float32, copy=False)  # d×N
> main
            TEXTS = texts
        elif tmp.endswith(".npz"):
            dat = np.load(tmp, allow_pickle=False)
            if "V" in dat:
                V = _ensure_float32_col_unit(np.array(dat["V"]))
                d, N = V.shape
                if "ids" in dat:
                    ids = list(map(str, dat["ids"]))
                else:
                    ids = [f"id{i}" for i in range(N)]
            elif "vectors" in dat:
                M = np.array(dat["vectors"]).astype(np.float32, copy=False)  # N×d
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
                V = M.T
                if "ids" in dat:
                    ids = list(map(str, dat["ids"]))
                else:
                    ids = [f"id{i}" for i in range(M.shape[0])]
            elif "X" in dat:
                M = np.array(dat["X"]).astype(np.float32, copy=False)
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
                V = M.T
                if "ids" in dat:
                    ids = list(map(str, dat["ids"]))
                else:
                    ids = [f"id{i}" for i in range(M.shape[0])]
            else:
                raise ValueError(".npz must contain 'V' (d×N) or 'vectors'/'X' (N×d)")
        else:
            raise ValueError("Unsupported file type. Use .jsonl or .npz")

        assert V is not None
        d, N = int(V.shape[0]), int(V.shape[1])
        # truncate/extend ids
        if len(ids) < N:
            ids += [f"id{i}" for i in range(len(ids), N)]
        elif len(ids) > N:
            ids = ids[:N]
        sid = uuid.uuid4().hex
        QSESS[sid] = {
            "V": V,
            "ids": ids,

            "texts": locals().get("TEXTS", []),

            "texts": locals().get("TEXTS", [])
> main
            "d": d,
            "N": N,
            "neighbors": None,
            "weights": None,
            "m_hat": None,
            "H_hat": None,
        }
        return JSONResponse({"sid": sid, "d": d, "N": N, "ids_head": ids[:5]})
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ---------- proxies ----------
@app.post("/qvnm/estimate_id")
async def proxy_estimate_id(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    req = {
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "k": int(payload.get("k", 10)),
        "gamma": float(payload.get("gamma", 0.5)),
        "alpha": float(payload.get("alpha", 0.5)),
        "boots": int(payload.get("boots", 8)),
        "mode": payload.get("mode", "local"),
    }
    if req["mode"] == "local":
        req["r"] = int(payload.get("r", 64))
    out = await _post_json(f"{JULIA_BASE}/qvnm/estimate_id", req)

    mode = out.get("mode", "local")
    if mode == "global":
        # expand scalars to vectors
        m_hat = float(out.get("m_hat", 0.0))
        H_hat = float(out.get("H_hat", 0.0))
        sess["m_hat"] = [m_hat] * N
        sess["H_hat"] = [H_hat] * N
    else:
        # ensure numeric lists and fill NaNs with means
        m_hat = np.array(out.get("m_hat", [0.0] * N), dtype=np.float64)
        H_hat = np.array(out.get("H_hat", [0.0] * N), dtype=np.float64)
        # replace non-finite
        for arr in (m_hat, H_hat):
            mask = ~np.isfinite(arr)
            if mask.any():
                arr[mask] = float(np.nanmean(arr[~mask])) if (~mask).any() else 0.0
        sess["m_hat"] = m_hat.astype(float).tolist()
        sess["H_hat"] = H_hat.astype(float).tolist()
    return JSONResponse({"sid": sid, **out})


@app.post("/qvnm/build_preview")
async def proxy_build_preview(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    # ensure estimates
    if not sess.get("m_hat") or not sess.get("H_hat"):
        return JSONResponse({"error": "run estimate_id first"}, status_code=400)
    # ensure knn graph
    k_graph = int(payload.get("knn_k", 10))
    if sess.get("neighbors") is None or sess.get("weights") is None or sess.get("_k_graph") != k_graph:
        nei, wts = _knn_graph_from_V(V, k=k_graph)
        sess["neighbors"], sess["weights"], sess["_k_graph"] = nei, wts, k_graph
    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "lambda_m": float(payload.get("lambda_m", 0.3)),
        "lambda_h": float(payload.get("lambda_h", 0.3)),
        "r": int(payload.get("r", 2)),
        "k_eval": int(payload.get("k_eval", 10)),
        "bins": int(payload.get("bins", 20)),
    }
    out = await _post_json(f"{JULIA_BASE}/qvnm/preview", req)
    # cache preview for exports
    sess["preview"] = out
    return JSONResponse({"sid": sid, **out})

    # cache preview for exports
    sess["preview"] = out
    return JSONResponse({"sid": sid, **out})


# ---------- Qwen optimize: retrieve context and propose diffs ----------
async def run_qwen(prompt: str, model_id: str | None = None, max_new_tokens: int = 800) -> str:
    """
    - If MODEL_ENDPOINT is set, POST there as a simple JSON API {prompt, max_new_tokens}
    - Else try local transformers using QWEN_MODEL or model_id.
    """
    endpoint = os.environ.get("MODEL_ENDPOINT")
    if endpoint:
        async with httpx.AsyncClient(timeout=300.0) as cx:
            r = await cx.post(endpoint, json={"prompt": prompt, "max_new_tokens": max_new_tokens})
            r.raise_for_status()
            try:
                j = r.json()
                return j.get("text") or j.get("output") or r.text
            except Exception:
                return r.text

    # Local transformers fallback
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    mid = model_id or os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    msgs = [
        {"role": "system", "content": "You are a senior engineer. Be concise; return unified diffs when changing code."},
        {"role": "user", "content": prompt},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


def build_prompt(goal: str, seed_id: str, ctx_blobs: list[dict]) -> str:
    header = (
        "You are given code/docs context. Task: propose concrete performance, safety, or readability optimizations relevant to the goal. Return:\n"
        "1) a brief rationale; 2) prioritized checklist; 3) unified diffs for files you modify; 4) tests if applicable.\n\n"
        f"GOAL:\n{goal}\n\n"
        f"SEED: {seed_id}\n\n"
        "CONTEXT (IDs + excerpts):\n"
    )
    parts: list[str] = []
    for blob in ctx_blobs:
        txt = str(blob.get("text", ""))[:3000]
        parts.append(f"--- {blob.get('id','')} ---\n{txt}\n")
    return header + "\n".join(parts)


@app.post("/pilot/optimize")
async def pilot_optimize(
    session: str,
    seed_id: str,
    goal: str = Body(..., embed=True),
    topk: int = 8,
    steps: int = 10,
    alpha: float = 0.85,
    theta: float = 0.0,
    model_id: str | None = None,
):
    sess = QSESS.get(session)
    if not sess:
        return JSONResponse({"error": "bad session"}, status_code=400)
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    # ensure graph and estimates
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=10)
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        sess["m_hat"], sess["H_hat"] = [0.0] * N, [0.0] * N

    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "ids": sess.get("ids"),
        "seed_id": seed_id,
        "topk": int(topk),
        "steps": int(steps),
        "alpha": float(alpha),
        "theta": float(theta),
    }
    ans = await _post_json(f"{JULIA_BASE}/qvnm/query", req)

    top_ids = [x.get("id") for x in (ans.get("top") or []) if x.get("id")]
    ctx = _texts_for_ids(sess, top_ids)
    prompt = build_prompt(goal, seed_id, ctx)
    llm_out = await run_qwen(prompt, model_id=model_id, max_new_tokens=1200)
    return JSONResponse({
        "seed": seed_id,
        "goal": goal,
        "top": ans.get("top"),
        "qwen": llm_out,
    })

# ---------- Qwen optimize: retrieve context and propose diffs ----------
async def run_qwen(prompt: str, model_id: str | None = None, max_new_tokens: int = 800) -> str:
    """
    - If MODEL_ENDPOINT is set, POST there as a simple JSON API {prompt, max_new_tokens}
    - Else try local transformers using QWEN_MODEL or model_id.
    """
    endpoint = os.environ.get("MODEL_ENDPOINT")
    if endpoint:
        async with httpx.AsyncClient(timeout=300.0) as cx:
            r = await cx.post(endpoint, json={"prompt": prompt, "max_new_tokens": max_new_tokens})
            r.raise_for_status()
            try:
                j = r.json()
                return j.get("text") or j.get("output") or r.text
            except Exception:
                return r.text

    # Local transformers fallback
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    mid = model_id or os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    msgs = [
        {"role": "system", "content": "You are a senior engineer. Be concise; return unified diffs when changing code."},
        {"role": "user", "content": prompt},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


def build_prompt(goal: str, seed_id: str, ctx_blobs: list[dict]) -> str:
    header = (
        "You are given code/docs context. Task: propose concrete performance, safety, or readability optimizations relevant to the goal. Return:\n"
        "1) a brief rationale; 2) prioritized checklist; 3) unified diffs for files you modify; 4) tests if applicable.\n\n"
        f"GOAL:\n{goal}\n\n"
        f"SEED: {seed_id}\n\n"
        "CONTEXT (IDs + excerpts):\n"
    )
    parts: list[str] = []
    for blob in ctx_blobs:
        txt = str(blob.get("text", ""))[:3000]
        parts.append(f"--- {blob.get('id','')} ---\n{txt}\n")
    return header + "\n".join(parts)


@app.post("/pilot/optimize")
async def pilot_optimize(
    session: str,
    seed_id: str,
    goal: str = Body(..., embed=True),
    topk: int = 8,
    steps: int = 10,
    alpha: float = 0.85,
    theta: float = 0.0,
    model_id: str | None = None,
):
    sess = QSESS.get(session)
    if not sess:
        return JSONResponse({"error": "bad session"}, status_code=400)
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    # ensure graph and estimates
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=10)
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        sess["m_hat"], sess["H_hat"] = [0.0] * N, [0.0] * N

    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "ids": sess.get("ids"),
        "seed_id": seed_id,
        "topk": int(topk),
        "steps": int(steps),
        "alpha": float(alpha),
        "theta": float(theta),
    }
    ans = await _post_json(f"{JULIA_BASE}/qvnm/query", req)

    top_ids = [x.get("id") for x in (ans.get("top") or []) if x.get("id")]
    ctx = _texts_for_ids(sess, top_ids)
    prompt = build_prompt(goal, seed_id, ctx)
    llm_out = await run_qwen(prompt, model_id=model_id, max_new_tokens=1200)
    return JSONResponse({
        "seed": seed_id,
        "goal": goal,
        "top": ans.get("top"),
        "qwen": llm_out,
    })

@app.post("/qvnm/query")
async def proxy_query(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=int(payload.get("knn_k", 10)))
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        # default to zeros if not estimated, to allow demo
        sess["m_hat"] = [0.0] * N
        sess["H_hat"] = [0.0] * N
    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "ids": sess["ids"],
        "seed_id": payload.get("seed_id"),
        "topk": int(payload.get("topk", 5)),
        "steps": int(payload.get("steps", 10)),
        "alpha": float(payload.get("alpha", 0.85)),
        "theta": float(payload.get("theta", 0.0)),
    }
    if "prior" in payload and payload["prior"] is not None:
        req["prior"] = payload["prior"]
    out = await _post_json(f"{JULIA_BASE}/qvnm/query", req)
    return JSONResponse({"sid": sid, **out})

> main
@app.post("/qvnm/query_traj")
async def proxy_query_traj(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=int(payload.get("knn_k", 10)))
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        sess["m_hat"] = [0.0] * N
        sess["H_hat"] = [0.0] * N
    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "ids": sess["ids"],
        "seed_id": payload.get("seed_id"),
        "steps": int(payload.get("steps", 20)),
        "alpha": float(payload.get("alpha", 0.85)),
        "theta": float(payload.get("theta", 0.0)),
    }
    if "prior" in payload and payload["prior"] is not None:
        req["prior"] = payload["prior"]
    out = await _post_json(f"{JULIA_BASE}/qvnm/query_traj", req)
    return JSONResponse({"sid": sid, **out})
> main
@app.post("/qvnm/build_codes")
async def proxy_build_codes(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=int(payload.get("knn_k", 10)))
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        sess["m_hat"] = [0.0] * N
        sess["H_hat"] = [0.0] * N
    req = {
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "lambda_code": float(payload.get("lambda_code", 0.25)),
        "hard": bool(payload.get("hard", False)),
    }
    out = await _post_json(f"{JULIA_BASE}/qvnm/build_codes", req)
    # cache codes summary for exports
    sess["codes"] = out.get("codes")
    return JSONResponse({"sid": sid, **out})

    # cache codes summary for exports
    sess["codes"] = out.get("codes")
    return JSONResponse({"sid": sid, **out})


# --- helpers to rebuild W and get coords from last preview ---
def _last_coords(sess: Dict[str, Any]) -> tuple[list[float], int]:
    prev = sess.get("preview") or {}
    em = prev.get("eigenmaps") or {}
    coords = em.get("coords") or []
    r = int(em.get("r", 0))
    return coords, r


async def _rebuild_W(sess: Dict[str, Any], k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3) -> list[list[float]]:
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    nei, wts = _knn_graph_from_V(V, k=k)
    req = {
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": nei,
        "weights": wts,
        "m_hat": sess.get("m_hat") or [0.0] * N,
        "H_hat": sess.get("H_hat") or [0.0] * N,
        "lambda_m": float(lambda_m),
        "lambda_h": float(lambda_h),
    }
    out = await _post_json(f"{JULIA_BASE}/qvnm/build", req)
    return out["W"]


# --- export as CSV/JSON files served by FastAPI ---
@app.get("/qvnm/export")
async def qvnm_export(session: str, kind: str = "coords_csv", k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3, threshold: float = 0.0):
    """
    kind ∈ {coords_csv, codes_csv, W_json, edges_csv}
    - coords_csv: N rows: node_id, x, y  (if r>=2; else returns 1D y=0)
    - codes_csv:  counts per code index (if available from last code-blend)
    - W_json:     dense adjacency as JSON (float32)
    - edges_csv:  i,j,weight (only weights >= threshold)
    """
    sid = session
    if sid not in QSESS:
        return JSONResponse({"error": "bad session"}, status_code=400)
    sess = QSESS[sid]
    ids = sess.get("ids") or [str(i) for i in range(sess["V"].shape[1])]

    if kind == "coords_csv":
        coords, r = _last_coords(sess)
        N = len(ids)
        xs = [0.0] * N
        ys = [0.0] * N
        if r >= 1 and coords and len(coords) >= r * N:
            # coords is vec(coords') in Julia ⇒ order: for i in 0..N-1, for d in 0..r-1: coords[i*r + d]
            for i in range(N):
                xs[i] = float(coords[i * r + 0])
                ys[i] = float(coords[i * r + 1]) if r >= 2 else 0.0
        # build CSV
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["node_id", "x", "y"])
        for nid, x, y in zip(ids, xs, ys):
            w.writerow([nid, f"{x:.6g}", f"{y:.6g}"])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    if kind == "codes_csv":
        codes = sess.get("codes") or {}
        hist = codes.get("hist") if isinstance(codes, dict) else None
        if hist is None:
            return JSONResponse({"error": "no codes available; run build_codes first"}, status_code=400)
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["code", "count"])
        for idx, cnt in enumerate(hist, start=1):
            w.writerow([idx, int(cnt)])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    if kind == "W_json":
        W = await _rebuild_W(sess, k=int(k), lambda_m=float(lambda_m), lambda_h=float(lambda_h))
        return JSONResponse({"W": W})

    if kind == "edges_csv":
        W = await _rebuild_W(sess, k=int(k), lambda_m=float(lambda_m), lambda_h=float(lambda_h))
        thr = float(threshold)
        N = len(W)
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["i", "j", "weight"])  # 1-based indices as in Julia
        for i in range(N):
            row = W[i]
            for j in range(i + 1, N):  # undirected edges once
                wij = float(row[j])
                if wij >= thr:
                    w.writerow([i + 1, j + 1, f"{wij:.6g}"])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    return JSONResponse({"error": "unknown kind"}, status_code=400)


# --- helpers to rebuild W and get coords from last preview ---
def _last_coords(sess: Dict[str, Any]) -> tuple[list[float], int]:
    prev = sess.get("preview") or {}
    em = prev.get("eigenmaps") or {}
    coords = em.get("coords") or []
    r = int(em.get("r", 0))
    return coords, r


async def _rebuild_W(sess: Dict[str, Any], k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3) -> list[list[float]]:
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    nei, wts = _knn_graph_from_V(V, k=k)
    req = {
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": nei,
        "weights": wts,
        "m_hat": sess.get("m_hat") or [0.0] * N,
        "H_hat": sess.get("H_hat") or [0.0] * N,
        "lambda_m": float(lambda_m),
        "lambda_h": float(lambda_h),
    }
    out = await _post_json(f"{JULIA_BASE}/qvnm/build", req)
    return out["W"]


# --- export as CSV/JSON files served by FastAPI ---
@app.get("/qvnm/export")
async def qvnm_export(session: str, kind: str = "coords_csv", k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3, threshold: float = 0.0):
    """
    kind ∈ {coords_csv, codes_csv, W_json, edges_csv}
    - coords_csv: N rows: node_id, x, y  (if r>=2; else returns 1D y=0)
    - codes_csv:  counts per code index (if available from last code-blend)
    - W_json:     dense adjacency as JSON (float32)
    - edges_csv:  i,j,weight (only weights >= threshold)
    """
    sid = session
    if sid not in QSESS:
        return JSONResponse({"error": "bad session"}, status_code=400)
    sess = QSESS[sid]
    ids = sess.get("ids") or [str(i) for i in range(sess["V"].shape[1])]

    if kind == "coords_csv":
        coords, r = _last_coords(sess)
        N = len(ids)
        xs = [0.0] * N
        ys = [0.0] * N
        if r >= 1 and coords and len(coords) >= r * N:
            # coords is vec(coords') in Julia ⇒ order: for i in 0..N-1, for d in 0..r-1: coords[i*r + d]
            for i in range(N):
                xs[i] = float(coords[i * r + 0])
                ys[i] = float(coords[i * r + 1]) if r >= 2 else 0.0
        # build CSV
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["node_id", "x", "y"])
        for nid, x, y in zip(ids, xs, ys):
            w.writerow([nid, f"{x:.6g}", f"{y:.6g}"])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    if kind == "codes_csv":
        codes = sess.get("codes") or {}
        hist = codes.get("hist") if isinstance(codes, dict) else None
        if hist is None:
            return JSONResponse({"error": "no codes available; run build_codes first"}, status_code=400)
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["code", "count"])
        for idx, cnt in enumerate(hist, start=1):
            w.writerow([idx, int(cnt)])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    if kind == "W_json":
        W = await _rebuild_W(sess, k=int(k), lambda_m=float(lambda_m), lambda_h=float(lambda_h))
        return JSONResponse({"W": W})

    if kind == "edges_csv":
        W = await _rebuild_W(sess, k=int(k), lambda_m=float(lambda_m), lambda_h=float(lambda_h))
        thr = float(threshold)
        N = len(W)
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["i", "j", "weight"])  # 1-based indices as in Julia
        for i in range(N):
            row = W[i]
            for j in range(i + 1, N):  # undirected edges once
                wij = float(row[j])
                if wij >= thr:
                    w.writerow([i + 1, j + 1, f"{wij:.6g}"])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    return JSONResponse({"error": "unknown kind"}, status_code=400)


@app.post("/cpl/init")
async def proxy_cpl_init(payload: Dict[str, Any]) -> JSONResponse:
    req = {
        "f": int(payload.get("f", 256)),
        "c": int(payload.get("c", 4096)),
        "tau": float(payload.get("tau", 0.07)),
        "seed": int(payload.get("seed", 2214)),
    }
    out = await _post_json(f"{JULIA_BASE}/cpl/init", req)
    return JSONResponse(out)


@app.get("/qvnm/session/{sid}")
async def get_session_meta(sid: str) -> JSONResponse:
    if sid not in QSESS:
        return JSONResponse({"error": "not found"}, status_code=404)
    sess = QSESS[sid]
    return JSONResponse({
        "sid": sid,
        "d": sess["d"],
        "N": sess["N"],
        "ids_head": sess["ids"][:5],
        "has_graph": sess.get("neighbors") is not None,
        "has_estimates": sess.get("m_hat") is not None,
    })


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)