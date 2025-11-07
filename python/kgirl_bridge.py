"""
kgirl_bridge.py - Integration bridge for kgirl ecosystem

This module provides integration with the kgirl repository's advanced cognitive systems:
- ChaosRAGJulia: Chaos router, HHT/EEMD time-frequency analytics, OpenAI RAG
- Recursive AI: Dynamic depth analysis with fractal resonance
- Holographic Memory: Associative recall and pattern recognition
- FAISS Vector Search: High-performance similarity search
- PostgreSQL pgvector: Persistent vector storage

Integration Architecture:
    ┌─────────────────────────────────────────────────────┐
    │              SHOUT + KGIRL INTEGRATION              │
    │                                                     │
    │  ┌──────────────┐           ┌──────────────┐      │
    │  │ CCL Analysis │──────────▶│ Chaos Router │      │
    │  │ (Ghosts)     │  entropy  │ (Market)     │      │
    │  └──────────────┘           └──────────────┘      │
    │                                     │               │
    │  ┌──────────────┐                  │               │
    │  │ QVNM Vectors │◀─────────────────┘               │
    │  │              │    chaos-adapted                 │
    │  └──────┬───────┘    retrieval                     │
    │         │                                           │
    │         ▼                                           │
    │  ┌──────────────┐           ┌──────────────┐      │
    │  │ Recursive AI │◀─────────▶│ Holographic  │      │
    │  │ Depth Search │  fractal  │ Memory       │      │
    │  └──────────────┘  resonance └──────────────┘      │
    │                                                     │
    │  ┌──────────────┐           ┌──────────────┐      │
    │  │ FAISS Search │◀─────────▶│ pgvector DB  │      │
    │  │ (Fast Query) │   hybrid  │ (Persistent) │      │
    │  └──────────────┘           └──────────────┘      │
    └─────────────────────────────────────────────────────┘
"""

import numpy as np
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class ChaosMode(Enum):
    """Chaos routing modes based on market/system stress levels."""
    CALM = "calm"  # Low chaos: precise retrieval
    MODERATE = "moderate"  # Medium chaos: balanced exploration
    TURBULENT = "turbulent"  # High chaos: diverse exploration
    CRITICAL = "critical"  # Extreme chaos: maximum diversity


@dataclass
class RecursiveDepthConfig:
    """Configuration for recursive cognitive depth analysis."""
    max_depth: int = 5
    branching_factor: int = 3
    resonance_threshold: float = 0.7
    fractal_dimension: float = 1.8


class KGirlBridge:
    """
    Bridge to kgirl's advanced cognitive systems.

    This class provides integration with:
    - ChaosRAG for adaptive retrieval under uncertainty
    - Recursive AI for deep cognitive analysis
    - Holographic memory for associative pattern matching
    - FAISS/pgvector for high-performance vector search
    """

    def __init__(
        self,
        kgirl_url: str = "http://localhost:9000",
        faiss_index_path: Optional[str] = None,
        postgres_connection: Optional[str] = None,
        enable_chaos: bool = True,
        enable_recursive: bool = True,
    ):
        """
        Initialize kgirl bridge.

        Args:
            kgirl_url: Base URL for kgirl services
            faiss_index_path: Path to FAISS index file (optional)
            postgres_connection: PostgreSQL connection string with pgvector
            enable_chaos: Enable chaos-adaptive routing
            enable_recursive: Enable recursive cognitive depth search
        """
        self.kgirl_url = kgirl_url
        self.faiss_index_path = faiss_index_path
        self.postgres_connection = postgres_connection
        self.enable_chaos = enable_chaos
        self.enable_recursive = enable_recursive

        # Initialize optional components
        self.faiss_index = None
        self.pg_conn = None

        if faiss_index_path:
            self._init_faiss()
        if postgres_connection:
            self._init_pgvector()

    def _init_faiss(self):
        """Initialize FAISS index for fast similarity search."""
        try:
            import faiss
            self.faiss_index = faiss.read_index(self.faiss_index_path)
            print(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")
        except ImportError:
            print("Warning: faiss-cpu not installed. Install with: pip install faiss-cpu")
        except Exception as e:
            print(f"Warning: Could not load FAISS index: {e}")

    def _init_pgvector(self):
        """Initialize PostgreSQL connection with pgvector extension."""
        try:
            import psycopg2
            self.pg_conn = psycopg2.connect(self.postgres_connection)
            print("PostgreSQL pgvector connection established")
        except ImportError:
            print("Warning: psycopg2 not installed. Install with: pip install psycopg2-binary")
        except Exception as e:
            print(f"Warning: Could not connect to PostgreSQL: {e}")

    def compute_chaos_metric(self, entropy_values: List[float], ghost_scores: List[float]) -> float:
        """
        Compute chaos metric from CCL analysis results.

        Higher chaos indicates more uncertainty/volatility in the codebase,
        which should trigger more exploratory retrieval strategies.

        Args:
            entropy_values: Shannon entropy values from CCL analysis
            ghost_scores: Ghost detection scores from CCL

        Returns:
            Chaos metric in [0, 1] where 1 is maximum chaos
        """
        if not entropy_values or not ghost_scores:
            return 0.5  # Default to moderate chaos

        # Normalize inputs
        entropy_arr = np.array(entropy_values)
        ghost_arr = np.array(ghost_scores)

        # Chaos components
        entropy_chaos = np.std(entropy_arr) / (np.mean(entropy_arr) + 1e-12)
        ghost_chaos = np.mean(ghost_arr)

        # Combined chaos metric (weighted average)
        chaos = 0.6 * entropy_chaos + 0.4 * ghost_chaos

        # Clamp to [0, 1]
        return float(np.clip(chaos, 0, 1))

    def determine_chaos_mode(self, chaos_metric: float) -> ChaosMode:
        """
        Determine chaos routing mode based on metric.

        Args:
            chaos_metric: Chaos value in [0, 1]

        Returns:
            Appropriate ChaosMode
        """
        if chaos_metric < 0.25:
            return ChaosMode.CALM
        elif chaos_metric < 0.5:
            return ChaosMode.MODERATE
        elif chaos_metric < 0.75:
            return ChaosMode.TURBULENT
        else:
            return ChaosMode.CRITICAL

    def chaos_rag_query(
        self,
        query_vector: List[float],
        chaos_mode: ChaosMode,
        top_k: int = 10,
        time_frequency_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute ChaosRAG query with adaptive retrieval strategy.

        Args:
            query_vector: Query embedding vector
            chaos_mode: Chaos routing mode
            top_k: Number of results to retrieve
            time_frequency_analysis: Enable HHT/EEMD analysis

        Returns:
            Retrieved results with chaos-adapted ranking
        """
        # Adjust retrieval parameters based on chaos mode
        if chaos_mode == ChaosMode.CALM:
            temperature = 0.1
            diversity_weight = 0.0
        elif chaos_mode == ChaosMode.MODERATE:
            temperature = 0.3
            diversity_weight = 0.2
        elif chaos_mode == ChaosMode.TURBULENT:
            temperature = 0.6
            diversity_weight = 0.5
        else:  # CRITICAL
            temperature = 1.0
            diversity_weight = 0.8

        # Build request payload
        payload = {
            "query_vector": query_vector,
            "top_k": top_k,
            "temperature": temperature,
            "diversity_weight": diversity_weight,
            "chaos_mode": chaos_mode.value,
            "enable_hht": time_frequency_analysis,
        }

        try:
            response = requests.post(
                f"{self.kgirl_url}/chaos_rag/query",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"ChaosRAG query failed: {e}")
            # Fallback to local FAISS search if available
            if self.faiss_index:
                return self._faiss_fallback_search(query_vector, top_k)
            return {"error": str(e), "results": []}

    def _faiss_fallback_search(self, query_vector: List[float], top_k: int) -> Dict[str, Any]:
        """Fallback search using local FAISS index."""
        if not self.faiss_index:
            return {"error": "No FAISS index available", "results": []}

        try:
            import faiss
            query_arr = np.array([query_vector], dtype=np.float32)
            distances, indices = self.faiss_index.search(query_arr, top_k)

            results = [
                {"id": int(idx), "distance": float(dist)}
                for idx, dist in zip(indices[0], distances[0])
            ]
            return {"results": results, "source": "faiss_fallback"}
        except Exception as e:
            return {"error": str(e), "results": []}

    def recursive_cognitive_search(
        self,
        initial_query: List[float],
        config: RecursiveDepthConfig,
        ccl_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform recursive cognitive depth search with fractal resonance.

        This method explores conceptual space at multiple levels of abstraction,
        generating variations and insights at each depth level.

        Args:
            initial_query: Initial query vector
            config: Recursive depth configuration
            ccl_report: Optional CCL analysis report for guidance

        Returns:
            Multi-level cognitive search results with fractal patterns
        """
        results = {
            "depth_levels": [],
            "fractal_patterns": [],
            "resonance_scores": [],
            "emergent_insights": [],
        }

        current_queries = [initial_query]

        for depth in range(config.max_depth):
            level_results = []

            for query in current_queries:
                # Query at current depth
                query_result = self._recursive_query_level(query, depth, config)
                level_results.append(query_result)

                # Check for resonance with CCL analysis
                if ccl_report:
                    resonance = self._compute_resonance(query_result, ccl_report)
                    results["resonance_scores"].append({
                        "depth": depth,
                        "resonance": resonance,
                    })

            results["depth_levels"].append({
                "depth": depth,
                "n_queries": len(current_queries),
                "results": level_results,
            })

            # Generate next level queries via fractal branching
            current_queries = self._fractal_branch(
                level_results,
                config.branching_factor,
                config.fractal_dimension,
            )

            # Early termination if resonance threshold met
            if results["resonance_scores"] and \
               results["resonance_scores"][-1]["resonance"] > config.resonance_threshold:
                results["emergent_insights"].append({
                    "depth": depth,
                    "insight": "High resonance achieved - convergence detected",
                })
                break

        # Analyze fractal patterns across depths
        results["fractal_patterns"] = self._analyze_fractal_patterns(results["depth_levels"])

        return results

    def _recursive_query_level(
        self,
        query: List[float],
        depth: int,
        config: RecursiveDepthConfig,
    ) -> Dict[str, Any]:
        """Execute query at specific recursive depth level."""
        try:
            payload = {
                "query_vector": query,
                "depth": depth,
                "branching_factor": config.branching_factor,
            }
            response = requests.post(
                f"{self.kgirl_url}/recursive_ai/query",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Fallback to local processing
            return {
                "depth": depth,
                "query_id": hash(tuple(query)) % 1000000,
                "variations": self._generate_local_variations(query, config.branching_factor),
                "source": "local_fallback",
            }

    def _generate_local_variations(self, query: List[float], n_variations: int) -> List[List[float]]:
        """Generate query variations locally using random perturbations."""
        query_arr = np.array(query)
        variations = []

        for _ in range(n_variations):
            # Add small random perturbation
            noise = np.random.normal(0, 0.1, size=query_arr.shape)
            variation = query_arr + noise
            # Normalize
            variation = variation / (np.linalg.norm(variation) + 1e-12)
            variations.append(variation.tolist())

        return variations

    def _compute_resonance(self, query_result: Dict[str, Any], ccl_report: Dict[str, Any]) -> float:
        """
        Compute resonance between query results and CCL analysis.

        Resonance indicates alignment between recursive search findings
        and static code analysis (CCL ghosts).
        """
        # Extract features from both sources
        # This is a simplified version - real implementation would be more sophisticated
        query_features = query_result.get("variations", [])
        ccl_ghosts = ccl_report.get("hotspots", [])

        if not query_features or not ccl_ghosts:
            return 0.5

        # Compute feature similarity (cosine similarity of aggregated features)
        # Simplified resonance computation
        resonance = 0.5 + 0.5 * np.random.random()  # Placeholder
        return float(resonance)

    def _fractal_branch(
        self,
        level_results: List[Dict[str, Any]],
        branching_factor: int,
        fractal_dimension: float,
    ) -> List[List[float]]:
        """
        Generate next level queries using fractal branching.

        Args:
            level_results: Results from current depth level
            branching_factor: Number of branches per node
            fractal_dimension: Fractal dimension for self-similar scaling

        Returns:
            List of query vectors for next depth level
        """
        next_queries = []

        for result in level_results:
            variations = result.get("variations", [])
            if not variations:
                continue

            # Select top variations based on fractal scaling
            n_select = min(branching_factor, len(variations))
            selected = variations[:n_select]
            next_queries.extend(selected)

        # Limit total queries using fractal dimension
        max_queries = int(branching_factor ** fractal_dimension)
        return next_queries[:max_queries]

    def _analyze_fractal_patterns(self, depth_levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze fractal patterns across recursive depth levels."""
        patterns = []

        for i, level in enumerate(depth_levels):
            n_queries = level["n_queries"]
            if i > 0:
                prev_n = depth_levels[i - 1]["n_queries"]
                growth_rate = n_queries / (prev_n + 1e-12)
            else:
                growth_rate = 1.0

            patterns.append({
                "depth": i,
                "n_queries": n_queries,
                "growth_rate": float(growth_rate),
                "self_similarity": self._compute_self_similarity(level),
            })

        return patterns

    def _compute_self_similarity(self, level_data: Dict[str, Any]) -> float:
        """Compute self-similarity metric for fractal analysis."""
        # Simplified self-similarity measure
        results = level_data.get("results", [])
        if len(results) < 2:
            return 1.0

        # Compute pairwise similarities between variations
        similarities = []
        for i in range(min(10, len(results))):
            for j in range(i + 1, min(10, len(results))):
                # Placeholder similarity computation
                sim = 0.7 + 0.3 * np.random.random()
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 1.0

    def holographic_memory_recall(
        self,
        pattern: List[float],
        association_depth: int = 3,
    ) -> Dict[str, Any]:
        """
        Perform holographic memory recall with associative pattern matching.

        Args:
            pattern: Input pattern vector
            association_depth: Depth of associative recall chain

        Returns:
            Associated memories and patterns
        """
        try:
            payload = {
                "pattern": pattern,
                "association_depth": association_depth,
            }
            response = requests.post(
                f"{self.kgirl_url}/holographic/recall",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": str(e),
                "associations": [],
            }

    def pgvector_persistent_store(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        table_name: str = "kgirl_vectors",
    ) -> Dict[str, Any]:
        """
        Store vectors in PostgreSQL with pgvector extension.

        Args:
            vectors: List of embedding vectors
            metadata: Associated metadata for each vector
            table_name: PostgreSQL table name

        Returns:
            Storage operation results
        """
        if not self.pg_conn:
            return {"error": "PostgreSQL connection not initialized"}

        try:
            cursor = self.pg_conn.cursor()

            # Create table if not exists
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    vector vector({len(vectors[0])}),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert vectors
            for vec, meta in zip(vectors, metadata):
                cursor.execute(
                    f"INSERT INTO {table_name} (vector, metadata) VALUES (%s, %s)",
                    (vec, json.dumps(meta))
                )

            self.pg_conn.commit()
            return {
                "stored": len(vectors),
                "table": table_name,
            }
        except Exception as e:
            self.pg_conn.rollback()
            return {"error": str(e)}

    def pgvector_similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        table_name: str = "kgirl_vectors",
    ) -> Dict[str, Any]:
        """
        Perform similarity search using pgvector.

        Args:
            query_vector: Query embedding
            top_k: Number of results
            table_name: PostgreSQL table name

        Returns:
            Search results
        """
        if not self.pg_conn:
            return {"error": "PostgreSQL connection not initialized"}

        try:
            cursor = self.pg_conn.cursor()

            # Query with cosine distance
            cursor.execute(f"""
                SELECT id, metadata, 1 - (vector <=> %s::vector) AS similarity
                FROM {table_name}
                ORDER BY vector <=> %s::vector
                LIMIT %s
            """, (query_vector, query_vector, top_k))

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "metadata": row[1],
                    "similarity": float(row[2]),
                })

            return {"results": results}
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# Integration Use Cases
# ============================================================================

def integrate_ccl_with_chaosrag(
    ccl_report_path: str,
    kgirl_bridge: KGirlBridge,
    query_embeddings: List[List[float]],
) -> Dict[str, Any]:
    """
    Integration use case: Use CCL analysis to drive chaos-adaptive retrieval.

    Args:
        ccl_report_path: Path to CCL JSON report
        kgirl_bridge: Initialized KGirlBridge instance
        query_embeddings: Query vectors for retrieval

    Returns:
        Integrated analysis results
    """
    # Load CCL report
    with open(ccl_report_path, 'r') as f:
        ccl_report = json.load(f)

    # Extract entropy and ghost metrics
    entropy_values = [func.get("entropy_bits", 0) for func in ccl_report.get("functions", [])]
    ghost_scores = [func.get("ghost_score", 0) for func in ccl_report.get("functions", [])]

    # Compute chaos metric
    chaos_metric = kgirl_bridge.compute_chaos_metric(entropy_values, ghost_scores)
    chaos_mode = kgirl_bridge.determine_chaos_mode(chaos_metric)

    print(f"Chaos metric: {chaos_metric:.3f} -> Mode: {chaos_mode.value}")

    # Execute chaos-adaptive queries
    results = []
    for query_vec in query_embeddings:
        result = kgirl_bridge.chaos_rag_query(
            query_vector=query_vec,
            chaos_mode=chaos_mode,
            top_k=10,
            time_frequency_analysis=True,
        )
        results.append(result)

    return {
        "chaos_metric": chaos_metric,
        "chaos_mode": chaos_mode.value,
        "ccl_summary": {
            "n_functions": len(ccl_report.get("functions", [])),
            "avg_entropy": float(np.mean(entropy_values)) if entropy_values else 0,
            "avg_ghost_score": float(np.mean(ghost_scores)) if ghost_scores else 0,
        },
        "retrieval_results": results,
    }
