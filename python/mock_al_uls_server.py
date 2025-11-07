"""
Minimal FastAPI mock for al-ULS /optimize so you can test the Julia bridge end-to-end.
Run:
  uvicorn python.mock_al_uls_server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np

app = FastAPI()


# ============================================================================
# OPTIMIZATION ALGORITHMS - Production-grade implementations
# ============================================================================

def _kfp_optimize(adj: np.ndarray, beta: float) -> tuple:
    """
    Kinetic Force Projection: Spectral method with stability constraints.
    Uses eigenvalue decomposition to find stable routing order.
    """
    n = adj.shape[0]
    # Symmetrize for stable eigendecomposition
    A_sym = (adj + adj.T) / 2.0

    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(A_sym)
        # Use dominant eigenvector for importance ranking
        dominant_vec = np.abs(eigenvectors[:, -1])
        # Apply beta-weighted combination with out-degree
        out_degree = np.sum(adj, axis=1)
        combined_score = beta * dominant_vec + (1 - beta) * (out_degree / (np.max(out_degree) + 1e-12))
        routing = np.argsort(-combined_score).tolist()
        objective = np.sum(combined_score * out_degree)
    except np.linalg.LinAlgError:
        # Fallback to degree-based routing
        out_degree = np.sum(adj, axis=1)
        routing = np.argsort(-out_degree).tolist()
        objective = np.sum(out_degree)

    return routing, objective


def _spectral_optimize(adj: np.ndarray) -> tuple:
    """
    Spectral graph partitioning using Fiedler vector.
    Finds natural cluster structure for optimal routing.
    """
    n = adj.shape[0]
    # Compute graph Laplacian
    D = np.diag(np.sum(adj, axis=1))
    L = D - adj

    try:
        # Find second-smallest eigenvector (Fiedler vector)
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]  # Second eigenvector
        # Sort by Fiedler vector to find natural ordering
        routing = np.argsort(fiedler).tolist()
        # Objective: minimize cut size
        objective = np.trace(adj) + np.sum(np.abs(fiedler))
    except np.linalg.LinAlgError:
        out_degree = np.sum(adj, axis=1)
        routing = np.argsort(-out_degree).tolist()
        objective = np.sum(out_degree)

    return routing, objective


def _pagerank_optimize(adj: np.ndarray, beta: float) -> tuple:
    """
    PageRank-based importance routing.
    Iteratively computes node importance based on incoming connections.
    """
    n = adj.shape[0]
    # Normalize adjacency matrix (column-stochastic)
    col_sums = np.sum(adj, axis=0) + 1e-12
    P = adj / col_sums

    # PageRank iteration
    alpha = beta  # Damping factor
    rank = np.ones(n) / n
    for _ in range(50):  # Max iterations
        rank_new = alpha * (P @ rank) + (1 - alpha) * np.ones(n) / n
        if np.linalg.norm(rank_new - rank) < 1e-6:
            break
        rank = rank_new

    routing = np.argsort(-rank).tolist()
    objective = np.sum(rank * np.sum(adj, axis=1))

    return routing, objective


def _entropy_optimize(adj: np.ndarray) -> tuple:
    """
    Maximum entropy routing: Balance information flow across network.
    Uses entropy-based metrics to find diverse routing paths.
    """
    n = adj.shape[0]

    # Compute node entropies based on connection distribution
    entropies = np.zeros(n)
    for i in range(n):
        out_weights = adj[i, :] + 1e-12
        probs = out_weights / np.sum(out_weights)
        entropies[i] = -np.sum(probs * np.log(probs + 1e-12))

    # High entropy nodes are prioritized for balanced routing
    routing = np.argsort(-entropies).tolist()
    objective = np.sum(entropies * np.sum(adj, axis=1))

    return routing, objective


def _hybrid_optimize(adj: np.ndarray, beta: float) -> tuple:
    """
    Hybrid approach: Combines multiple methods with adaptive weighting.
    Selects best strategy based on graph properties.
    """
    n = adj.shape[0]
    density = np.sum(adj > 0) / (n * n)

    # Choose strategy based on graph density
    if density > 0.5:
        # Dense graph: use spectral method
        return _spectral_optimize(adj)
    elif density > 0.1:
        # Medium density: use PageRank
        return _pagerank_optimize(adj, beta)
    else:
        # Sparse graph: use KFP
        return _kfp_optimize(adj, beta)


# ============================================================================
# ADVANCED MATRIX OPERATIONS
# ============================================================================

def _optimize_matrix_enhanced(matrix: List[List[float]], method: str = "sparsity") -> Dict[str, Any]:
    """
    Enhanced matrix optimization with multiple compression strategies.

    Methods:
    - sparsity: Threshold-based sparsification
    - svd_lowrank: SVD low-rank approximation
    - spectral_filter: Eigenvalue filtering
    - adaptive: Choose best method based on matrix properties
    """
    A = np.array(matrix, dtype=float)

    if method == "adaptive":
        # Choose method based on matrix characteristics
        singular_vals = np.linalg.svd(A, compute_uv=False)
        rank_deficit = np.sum(singular_vals > 1e-6) / len(singular_vals)
        method = "svd_lowrank" if rank_deficit < 0.5 else "sparsity"

    if method == "sparsity":
        # Threshold small values to zero at the 60th percentile
        thresh = np.percentile(np.abs(A), 60)
        B = A.copy()
        B[np.abs(B) < max(thresh, 1e-8)] = 0.0
    elif method == "svd_lowrank":
        # Keep top-k components
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        k = int(max(1, min(min(A.shape), 8)))
        B = (U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :])
    elif method == "spectral_filter":
        # Filter based on spectral properties
        if A.shape[0] == A.shape[1]:
            eigenvalues, eigenvectors = np.linalg.eigh((A + A.T) / 2)
            # Keep top 50% of eigenvalue energy
            energy = np.cumsum(np.abs(eigenvalues[::-1])) / np.sum(np.abs(eigenvalues))
            k = np.searchsorted(energy, 0.5) + 1
            B = eigenvectors[:, -k:] @ np.diag(eigenvalues[-k:]) @ eigenvectors[:, -k:].T
        else:
            # Fallback to SVD for non-square matrices
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            k = int(max(1, min(min(A.shape), 8)))
            B = (U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :])
    else:
        B = A

    zeros = np.sum(np.abs(B) < 1e-12)
    total = B.size
    reconstruction_error = np.linalg.norm(A - B, 'fro') / (np.linalg.norm(A, 'fro') + 1e-12)

    return {
        "optimized_matrix": B.tolist(),
        "compression_ratio": float(zeros) / float(total) if total else 0.0,
        "reconstruction_error": float(reconstruction_error),
        "method": method,
        "original_norm": float(np.linalg.norm(A, 'fro')),
        "compressed_norm": float(np.linalg.norm(B, 'fro')),
    }


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
    """
    Production-grade optimization using multiple strategies based on mode.

    Modes:
    - kfp: Kinetic Force Projection with spectral analysis
    - spectral: Spectral graph partitioning with eigenvector decomposition
    - pagerank: PageRank-based importance ordering
    - entropy: Maximum entropy routing
    - hybrid: Adaptive combination of methods
    """
    adj = np.array(inp.adjacency, dtype=float)
    n = len(adj)
    mode = inp.mode or "kfp"
    beta = inp.beta or 0.8

    if mode == "kfp":
        routing, obj = _kfp_optimize(adj, beta)
    elif mode == "spectral":
        routing, obj = _spectral_optimize(adj)
    elif mode == "pagerank":
        routing, obj = _pagerank_optimize(adj, beta)
    elif mode == "entropy":
        routing, obj = _entropy_optimize(adj)
    elif mode == "hybrid":
        routing, obj = _hybrid_optimize(adj, beta)
    else:
        # Fallback to simple sum-based routing
        out_sums = [(i, sum(adj[i])) for i in range(n)]
        routing = [i for i, _ in sorted(out_sums, key=lambda t: t[1], reverse=True)]
        obj = sum(sum(row) for row in adj)

    return {
        "routing": routing,
        "objective": float(obj),
        "mode": mode,
        "beta": beta,
        "metadata": {
            "n_nodes": n,
            "sparsity": float(np.sum(adj == 0)) / float(adj.size),
            "symmetry": float(np.linalg.norm(adj - adj.T)) / (float(np.linalg.norm(adj)) + 1e-12)
        }
    }


# Generic function-dispatch endpoint used by TA ULS trainer's JuliaClient
class FunctionCall(BaseModel):
    function: str
    args: List[Any]


def _optimize_matrix(matrix: List[List[float]], method: str = "sparsity") -> Dict[str, Any]:
    A = np.array(matrix, dtype=float)
    if method == "sparsity":
        # Threshold small values to zero at the 60th percentile of absolute values
        thresh = np.percentile(np.abs(A), 60)
        B = A.copy()
        B[np.abs(B) < max(thresh, 1e-8)] = 0.0
    elif method == "svd_lowrank":
        # Keep top-k components (k=min( min(A.shape), 8 ))
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        k = int(max(1, min(min(A.shape), 8)))
        S = np.zeros_like(A)
        np.fill_diagonal(S, s)
        B = (U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :])
    else:
        B = A
    zeros = np.sum(B == 0.0)
    total = B.size
    return {
        "optimized_matrix": B.tolist(),
        "compression_ratio": float(zeros) / float(total) if total else 0.0,
        "method": method,
    }


def _create_polynomials(data: List[List[float]], variables: List[str]) -> Dict[str, Any]:
    X = np.array(data, dtype=float)  # shape: (n_samples, n_features)
    n, d = X.shape
    vars_ = variables if variables else [f"x{i}" for i in range(d)]
    # Build polynomial features up to degree 2: 1, xi, xi*xj
    feats = [np.ones((n, 1))]
    names = ["1"]
    # Linear terms
    for j in range(d):
        feats.append(X[:, [j]])
        names.append(vars_[j])
    # Quadratic/cross terms
    for i in range(d):
        for j in range(i, d):
            feats.append((X[:, [i]] * X[:, [j]]))
            names.append(f"{vars_[i]}*{vars_[j]}")
    Phi = np.concatenate(feats, axis=1)
    # Return the design matrix summary and feature names
    return {
        "feature_names": names,
        "num_features": int(Phi.shape[1]),
        "num_samples": int(Phi.shape[0]),
        "feature_stats": {
            "mean": Phi.mean(axis=0).tolist(),
            "std": (Phi.std(axis=0) + 1e-12).tolist(),
        },
    }


def _analyze_polynomials(polynomials: Dict[str, Any]) -> Dict[str, Any]:
    means = np.array(polynomials.get("feature_stats", {}).get("mean", []), dtype=float)
    stds = np.array(polynomials.get("feature_stats", {}).get("std", []), dtype=float)
    return {
        "num_features": polynomials.get("num_features", 0),
        "dominant_features": int(np.sum(means > (means.mean() + stds.mean()))),
        "mean_of_means": float(means.mean()) if means.size else 0.0,
        "mean_of_stds": float(stds.mean()) if stds.size else 0.0,
    }


def _compute_intrinsic_dim(vectors: List[List[float]], k: int = 5) -> Dict[str, Any]:
    """
    Estimate intrinsic dimensionality using k-NN distance growth rate (Costa-Hero method).
    Complements the QVNM system's intrinsic dimension estimation.
    """
    X = np.array(vectors, dtype=float)
    n, d = X.shape

    if n < k + 1:
        return {"error": "Not enough samples for k-NN estimation", "k": k, "n_samples": n}

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(X, X, metric='euclidean')

    # For each point, find k-nearest neighbors (excluding self)
    knn_dists = []
    for i in range(n):
        sorted_dists = np.sort(dists[i, :])
        knn_dists.append(sorted_dists[1:k+1])  # Exclude self (distance 0)

    knn_dists = np.array(knn_dists)

    # Estimate intrinsic dimension via log-distance ratio
    # ID ≈ d log(r_k / r_1) / log(k)
    r1 = knn_dists[:, 0] + 1e-12
    rk = knn_dists[:, -1] + 1e-12
    local_ids = np.log(rk / r1) / (np.log(k) + 1e-12)

    # Robust estimate: use median to avoid outliers
    global_id = np.median(local_ids)
    id_std = np.std(local_ids)

    return {
        "intrinsic_dimension": float(global_id),
        "id_std": float(id_std),
        "ambient_dimension": int(d),
        "n_samples": int(n),
        "k": k,
        "local_ids": local_ids.tolist(),
        "compression_potential": float(1.0 - global_id / d) if d > 0 else 0.0,
    }


def _graph_coherence(adjacency: List[List[float]]) -> Dict[str, Any]:
    """
    Compute graph coherence metrics for network analysis.
    Measures connectivity, clustering, and spectral properties.
    """
    A = np.array(adjacency, dtype=float)
    n = A.shape[0]

    # Basic graph metrics
    density = np.sum(A > 0) / (n * n)
    out_degrees = np.sum(A, axis=1)
    in_degrees = np.sum(A, axis=0)

    # Spectral properties
    A_sym = (A + A.T) / 2.0
    try:
        eigenvalues = np.linalg.eigvalsh(A_sym)
        spectral_gap = float(eigenvalues[-1] - eigenvalues[-2]) if len(eigenvalues) > 1 else 0.0
        spectral_radius = float(np.max(np.abs(eigenvalues)))

        # Algebraic connectivity (second-smallest eigenvalue of Laplacian)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        laplacian_eigs = np.linalg.eigvalsh(L)
        algebraic_connectivity = float(laplacian_eigs[1]) if len(laplacian_eigs) > 1 else 0.0
    except np.linalg.LinAlgError:
        spectral_gap = 0.0
        spectral_radius = 0.0
        algebraic_connectivity = 0.0

    # Clustering coefficient (local transitivity)
    clustering_coeffs = []
    for i in range(n):
        neighbors = np.where(A[i, :] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering_coeffs.append(0.0)
        else:
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for m in range(j + 1, len(neighbors)):
                    if A[neighbors[j], neighbors[m]] > 0:
                        triangles += 1
            clustering_coeffs.append(2.0 * triangles / (k * (k - 1)))

    avg_clustering = float(np.mean(clustering_coeffs))

    return {
        "n_nodes": n,
        "density": float(density),
        "avg_out_degree": float(np.mean(out_degrees)),
        "avg_in_degree": float(np.mean(in_degrees)),
        "degree_std": float(np.std(out_degrees)),
        "spectral_gap": spectral_gap,
        "spectral_radius": spectral_radius,
        "algebraic_connectivity": algebraic_connectivity,
        "avg_clustering": avg_clustering,
        "coherence_score": float((spectral_gap + algebraic_connectivity) / (spectral_radius + 1e-12)),
    }


@app.post("/")
def function_dispatch(call: FunctionCall):
    """
    Generic function dispatch endpoint for Julia client compatibility.

    Supported functions:
    - optimize_matrix: Matrix compression and optimization
    - create_polynomials: Feature engineering with polynomial expansion
    - analyze_polynomials: Statistical analysis of polynomial features
    - compute_intrinsic_dim: Estimate intrinsic dimensionality (NEW)
    - graph_coherence: Compute graph coherence metrics (NEW)
    """
    if call.function == "optimize_matrix":
        matrix = call.args[0] if len(call.args) > 0 else []
        method = call.args[1] if len(call.args) > 1 else "sparsity"
        return _optimize_matrix_enhanced(matrix, method)
    if call.function == "create_polynomials":
        data = call.args[0] if len(call.args) > 0 else []
        variables = call.args[1] if len(call.args) > 1 else []
        return _create_polynomials(data, variables)
    if call.function == "analyze_polynomials":
        polys = call.args[0] if len(call.args) > 0 else {}
        return _analyze_polynomials(polys)
    if call.function == "compute_intrinsic_dim":
        vectors = call.args[0] if len(call.args) > 0 else []
        k = call.args[1] if len(call.args) > 1 else 5
        return _compute_intrinsic_dim(vectors, k)
    if call.function == "graph_coherence":
        adjacency = call.args[0] if len(call.args) > 0 else []
        return _graph_coherence(adjacency)
    return {"error": f"Unknown function: {call.function}"}