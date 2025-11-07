"""
unified_integration.py - Unified workflows integrating SHOUT, kgirl, and orwells-egg

This module demonstrates end-to-end workflows that integrate all three ecosystems:

1. SHOUT: CCL analysis, QVNM vectors, TA-ULS training, LIMPS bridge
2. kgirl: ChaosRAG, recursive AI, holographic memory, FAISS/pgvector
3. orwells-egg: RFV versioning, job queue, SQL generation, ML2 training

Complete System Architecture:
    ┌───────────────────────────────────────────────────────────────────┐
    │                    UNIFIED COGNITIVE SYSTEM                       │
    │                                                                   │
    │  ┌────────────┐    ┌────────────┐    ┌────────────────────┐    │
    │  │   SHOUT    │───▶│   kgirl    │───▶│   orwells-egg      │    │
    │  │            │    │            │    │                    │    │
    │  │ CCL Ghost  │    │ ChaosRAG   │    │ RFV Versioning     │    │
    │  │ Detection  │    │ Recursive  │    │ Job Queue          │    │
    │  │            │    │ Cognition  │    │ ML2 Training       │    │
    │  │ QVNM       │    │ Holographic│    │ SQL Generation     │    │
    │  │ Vectors    │    │ Memory     │    │                    │    │
    │  │            │    │            │    │                    │    │
    │  │ TA-ULS     │    │ FAISS/     │    │ Workflow           │    │
    │  │ Training   │    │ pgvector   │    │ Orchestration      │    │
    │  └────────────┘    └────────────┘    └────────────────────┘    │
    │         │                 │                      │               │
    │         └─────────────────┴──────────────────────┘               │
    │                           │                                      │
    │                  ┌────────▼────────┐                             │
    │                  │  Emergent        │                             │
    │                  │  Intelligence    │                             │
    │                  │  Layer          │                             │
    │                  └─────────────────┘                             │
    └───────────────────────────────────────────────────────────────────┘
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from kgirl_bridge import (
    KGirlBridge,
    ChaosMode,
    RecursiveDepthConfig,
    integrate_ccl_with_chaosrag,
)
from orwells_bridge import (
    OrwellsBridge,
    ArtifactType,
    JobPriority,
    integrate_ccl_with_rfv,
    integrate_tauls_with_job_queue,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Workflow 1: Chaos-Aware Code Analysis Pipeline
# ============================================================================

class ChaosAwareCodeAnalysisPipeline:
    """
    End-to-end pipeline: CCL → Chaos Detection → Adaptive RAG → RFV Versioning

    This workflow analyzes code for behavioral anomalies (ghosts), computes
    chaos metrics, performs chaos-adaptive retrieval, and versions results.
    """

    def __init__(
        self,
        kgirl_bridge: KGirlBridge,
        orwells_bridge: OrwellsBridge,
    ):
        self.kgirl = kgirl_bridge
        self.orwells = orwells_bridge

    def run(
        self,
        ccl_report_path: str,
        query_embeddings: List[List[float]],
    ) -> Dict[str, Any]:
        """
        Execute chaos-aware code analysis pipeline.

        Args:
            ccl_report_path: Path to CCL analysis report
            query_embeddings: Query vectors for semantic search

        Returns:
            Comprehensive analysis results
        """
        logger.info("Starting Chaos-Aware Code Analysis Pipeline")

        # Step 1: Load CCL analysis
        with open(ccl_report_path, 'r') as f:
            ccl_report = json.load(f)

        logger.info(f"Loaded CCL report: {len(ccl_report.get('functions', []))} functions")

        # Step 2: Extract entropy and ghost metrics
        functions = ccl_report.get("functions", [])
        entropy_values = [f.get("entropy_bits", 0) for f in functions]
        ghost_scores = [f.get("ghost_score", 0) for f in functions]

        # Step 3: Compute chaos metric
        chaos_metric = self.kgirl.compute_chaos_metric(entropy_values, ghost_scores)
        chaos_mode = self.kgirl.determine_chaos_mode(chaos_metric)

        logger.info(f"Chaos metric: {chaos_metric:.3f} → Mode: {chaos_mode.value}")

        # Step 4: Perform chaos-adaptive retrieval
        retrieval_results = []
        for i, query_vec in enumerate(query_embeddings):
            result = self.kgirl.chaos_rag_query(
                query_vector=query_vec,
                chaos_mode=chaos_mode,
                top_k=10,
                time_frequency_analysis=True,
            )
            retrieval_results.append(result)
            logger.info(f"Query {i+1}/{len(query_embeddings)} completed")

        # Step 5: Create RFV snapshots for versioning
        rfv_results = integrate_ccl_with_rfv(
            ccl_report_path=ccl_report_path,
            orwells_bridge=self.orwells,
            vector_embeddings=query_embeddings,
        )

        logger.info(f"Created {rfv_results['n_snapshots']} RFV snapshots")

        # Step 6: Store results in holographic memory
        for result in retrieval_results:
            if "results" in result:
                # Store top results in holographic memory for associative recall
                for item in result["results"][:3]:
                    if "vector" in item:
                        self.kgirl.holographic_memory_recall(
                            pattern=item["vector"],
                            association_depth=2,
                        )

        return {
            "chaos_metric": chaos_metric,
            "chaos_mode": chaos_mode.value,
            "retrieval_results": retrieval_results,
            "rfv_snapshots": rfv_results,
            "summary": {
                "n_functions": len(functions),
                "avg_entropy": float(np.mean(entropy_values)) if entropy_values else 0,
                "avg_ghost_score": float(np.mean(ghost_scores)) if ghost_scores else 0,
                "n_queries": len(query_embeddings),
            }
        }


# ============================================================================
# Workflow 2: Recursive Training Optimization
# ============================================================================

class RecursiveTrainingOptimization:
    """
    End-to-end pipeline: TA-ULS → Recursive Search → Job Queue → ML2 Training

    This workflow combines TA-ULS stability-aware training with recursive
    cognitive search to explore hyperparameter space, using job queues
    for distributed execution and ML2 coach policies.
    """

    def __init__(
        self,
        kgirl_bridge: KGirlBridge,
        orwells_bridge: OrwellsBridge,
    ):
        self.kgirl = kgirl_bridge
        self.orwells = orwells_bridge

    def run(
        self,
        initial_config: Dict[str, Any],
        parameter_embeddings: List[List[float]],
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute recursive training optimization.

        Args:
            initial_config: Initial TA-ULS training configuration
            parameter_embeddings: Vector embeddings of hyperparameters
            max_depth: Maximum recursive depth

        Returns:
            Optimization results
        """
        logger.info("Starting Recursive Training Optimization")

        # Step 1: Configure recursive search
        recursive_config = RecursiveDepthConfig(
            max_depth=max_depth,
            branching_factor=3,
            resonance_threshold=0.75,
            fractal_dimension=1.6,
        )

        # Step 2: Perform recursive hyperparameter search
        logger.info("Executing recursive cognitive search")
        recursive_results = self.kgirl.recursive_cognitive_search(
            initial_query=parameter_embeddings[0],
            config=recursive_config,
            ccl_report=None,
        )

        # Step 3: Extract promising configurations from recursive search
        training_configs = []
        for level in recursive_results["depth_levels"]:
            for result in level["results"][:2]:  # Top 2 per level
                config = initial_config.copy()
                # Modify config based on recursive variations
                # (simplified - real implementation would decode variations)
                config["learning_rate"] = 0.001 * (1.0 + 0.1 * level["depth"])
                config["batch_size"] = 32 * (2 ** level["depth"])
                training_configs.append(config)

        logger.info(f"Generated {len(training_configs)} training configurations")

        # Step 4: Submit training jobs to adaptive queue
        job_results = integrate_tauls_with_job_queue(
            orwells_bridge=self.orwells,
            training_configs=training_configs,
        )

        # Step 5: Create workflow for coordinated execution
        workflow_steps = [
            {
                "step_id": i,
                "type": "training",
                "config": config,
                "priority": "high" if i < 3 else "normal",
            }
            for i, config in enumerate(training_configs)
        ]

        workflow_id = self.orwells.create_workflow(
            workflow_name="recursive_training_optimization",
            steps=workflow_steps,
        )

        logger.info(f"Created workflow: {workflow_id}")

        return {
            "recursive_search": recursive_results,
            "n_configs": len(training_configs),
            "job_submissions": job_results,
            "workflow_id": workflow_id,
            "fractal_patterns": recursive_results["fractal_patterns"],
        }


# ============================================================================
# Workflow 3: Semantic Query to SQL with Vector Grounding
# ============================================================================

class SemanticQueryToSQL:
    """
    End-to-end pipeline: QVNM Semantic → Natural Language → SQL Generation

    This workflow uses QVNM's semantic understanding to enhance natural
    language to SQL conversion, grounding queries in vector space.
    """

    def __init__(
        self,
        kgirl_bridge: KGirlBridge,
        orwells_bridge: OrwellsBridge,
    ):
        self.kgirl = kgirl_bridge
        self.orwells = orwells_bridge

    def run(
        self,
        natural_language_query: str,
        vector_context: List[List[float]],
        schema_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert natural language to SQL with semantic grounding.

        Args:
            natural_language_query: Query in natural language
            vector_context: Vector embeddings for semantic context
            schema_info: Database schema information

        Returns:
            Generated SQL with semantic analysis
        """
        logger.info("Starting Semantic Query to SQL Pipeline")

        # Step 1: Compute intrinsic dimension of query space
        if len(vector_context) > 5:
            id_result = self._compute_query_space_structure(vector_context)
            logger.info(f"Query space ID: {id_result.get('intrinsic_dimension', 'N/A')}")
        else:
            id_result = {}

        # Step 2: Use holographic memory for schema recall
        schema_patterns = []
        if schema_info and "tables" in schema_info:
            for table in schema_info["tables"]:
                if "embedding" in table:
                    memory = self.kgirl.holographic_memory_recall(
                        pattern=table["embedding"],
                        association_depth=2,
                    )
                    schema_patterns.append(memory)

        # Step 3: Enhance query with semantic context
        qvnm_context = {
            "intrinsic_dimension": id_result.get("intrinsic_dimension", None),
            "schema_patterns": schema_patterns,
            "schema": schema_info,
        }

        # Step 4: Generate SQL using orwells-egg
        sql_result = self.orwells.generate_sql(
            natural_language_query=natural_language_query,
            schema_context=qvnm_context,
        )

        logger.info(f"Generated SQL: {sql_result.get('sql', 'N/A')[:100]}...")

        # Step 5: Create RFV snapshot of query
        if "sql" in sql_result and vector_context:
            snapshot = self.orwells.create_rfv_snapshot(
                artifact_type=ArtifactType.DATA,
                vector_embedding=vector_context[0],
                metadata={
                    "query_type": "semantic_sql",
                    "natural_language": natural_language_query,
                    "generated_sql": sql_result["sql"],
                    "intrinsic_dimension": id_result.get("intrinsic_dimension"),
                },
            )
            logger.info(f"Created RFV snapshot: {snapshot.snapshot_id}")

        return {
            "natural_language_query": natural_language_query,
            "sql_result": sql_result,
            "semantic_analysis": {
                "intrinsic_dimension": id_result.get("intrinsic_dimension"),
                "n_schema_patterns": len(schema_patterns),
            },
        }

    def _compute_query_space_structure(self, vectors: List[List[float]]) -> Dict[str, Any]:
        """Compute intrinsic structure of query vector space."""
        # Use mock_al_uls_server's compute_intrinsic_dim function
        import requests
        try:
            response = requests.post(
                "http://localhost:8000/",
                json={"function": "compute_intrinsic_dim", "args": [vectors, 5]},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Could not compute intrinsic dimension: {e}")
            return {}


# ============================================================================
# Workflow 4: Multi-System Self-Organizing Intelligence
# ============================================================================

class SelfOrganizingIntelligence:
    """
    Meta-workflow: Autonomous system that coordinates all three ecosystems.

    This is an emergent intelligence layer that:
    - Monitors system state across all three platforms
    - Adapts behavior based on chaos/entropy metrics
    - Self-organizes workflows based on observed patterns
    - Evolves strategies through recursive learning
    """

    def __init__(
        self,
        kgirl_bridge: KGirlBridge,
        orwells_bridge: OrwellsBridge,
        learning_rate: float = 0.1,
    ):
        self.kgirl = kgirl_bridge
        self.orwells = orwells_bridge
        self.learning_rate = learning_rate

        # System state
        self.chaos_history = []
        self.performance_history = []
        self.strategy_weights = {
            "chaos_aware_analysis": 0.33,
            "recursive_optimization": 0.33,
            "semantic_query": 0.33,
        }

    def observe_system_state(
        self,
        ccl_report: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        query_performance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Observe current state across all systems.

        Args:
            ccl_report: Latest CCL analysis
            training_metrics: Latest training metrics
            query_performance: Latest query performance

        Returns:
            Aggregated system state
        """
        state = {
            "timestamp": np.datetime64('now').astype(str),
            "chaos_level": 0.5,
            "training_stability": 0.5,
            "query_efficiency": 0.5,
        }

        # Compute chaos level from CCL
        if ccl_report:
            functions = ccl_report.get("functions", [])
            entropy_values = [f.get("entropy_bits", 0) for f in functions]
            ghost_scores = [f.get("ghost_score", 0) for f in functions]
            state["chaos_level"] = self.kgirl.compute_chaos_metric(
                entropy_values, ghost_scores
            )

        # Extract training stability
        if training_metrics:
            loss_variance = training_metrics.get("loss_variance", 1.0)
            state["training_stability"] = 1.0 / (1.0 + loss_variance)

        # Extract query efficiency
        if query_performance:
            avg_latency = query_performance.get("avg_latency_ms", 100)
            state["query_efficiency"] = 1.0 / (1.0 + avg_latency / 100.0)

        self.chaos_history.append(state["chaos_level"])
        self.performance_history.append(
            state["training_stability"] * state["query_efficiency"]
        )

        return state

    def adapt_strategy(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt strategy weights based on system state.

        Uses reinforcement learning principles to adjust strategy mix.

        Args:
            system_state: Current system state

        Returns:
            Updated strategy weights
        """
        chaos = system_state["chaos_level"]
        stability = system_state["training_stability"]
        efficiency = system_state["query_efficiency"]

        # Adjust weights based on system state
        # High chaos → more chaos_aware_analysis
        # Low stability → more recursive_optimization
        # Low efficiency → more semantic_query optimization

        target_weights = {
            "chaos_aware_analysis": 0.2 + 0.6 * chaos,
            "recursive_optimization": 0.2 + 0.6 * (1.0 - stability),
            "semantic_query": 0.2 + 0.6 * (1.0 - efficiency),
        }

        # Normalize
        total = sum(target_weights.values())
        target_weights = {k: v / total for k, v in target_weights.items()}

        # Smooth update using learning rate
        for key in self.strategy_weights:
            self.strategy_weights[key] = (
                (1 - self.learning_rate) * self.strategy_weights[key] +
                self.learning_rate * target_weights[key]
            )

        logger.info(f"Strategy weights adapted: {self.strategy_weights}")

        return self.strategy_weights

    def self_organize(
        self,
        n_iterations: int = 10,
        ccl_report_path: Optional[str] = None,
        vector_data: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute self-organizing intelligence loop.

        Args:
            n_iterations: Number of adaptation iterations
            ccl_report_path: Optional CCL report for monitoring
            vector_data: Optional vector data for analysis

        Returns:
            Self-organization results
        """
        logger.info(f"Starting Self-Organizing Intelligence ({n_iterations} iterations)")

        iteration_results = []

        for i in range(n_iterations):
            logger.info(f"Iteration {i+1}/{n_iterations}")

            # Step 1: Observe system state
            ccl_report = None
            if ccl_report_path and Path(ccl_report_path).exists():
                with open(ccl_report_path, 'r') as f:
                    ccl_report = json.load(f)

            system_state = self.observe_system_state(
                ccl_report=ccl_report,
                training_metrics={"loss_variance": np.random.random()},  # Placeholder
                query_performance={"avg_latency_ms": 50 + 50 * np.random.random()},
            )

            # Step 2: Adapt strategy
            strategy_weights = self.adapt_strategy(system_state)

            # Step 3: Execute weighted combination of workflows
            workflow_results = {}

            if vector_data and len(vector_data) > 0:
                # Sample workflows based on weights
                if np.random.random() < strategy_weights["chaos_aware_analysis"]:
                    logger.info("Executing chaos-aware analysis")
                    # Simplified execution
                    workflow_results["chaos_aware"] = {
                        "executed": True,
                        "chaos_metric": system_state["chaos_level"],
                    }

                if np.random.random() < strategy_weights["recursive_optimization"]:
                    logger.info("Executing recursive optimization")
                    workflow_results["recursive_opt"] = {
                        "executed": True,
                        "stability": system_state["training_stability"],
                    }

                if np.random.random() < strategy_weights["semantic_query"]:
                    logger.info("Executing semantic query optimization")
                    workflow_results["semantic_query"] = {
                        "executed": True,
                        "efficiency": system_state["query_efficiency"],
                    }

            iteration_results.append({
                "iteration": i,
                "system_state": system_state,
                "strategy_weights": strategy_weights.copy(),
                "workflows_executed": workflow_results,
            })

        # Analyze emergent patterns
        emergent_patterns = self._analyze_emergence(iteration_results)

        return {
            "n_iterations": n_iterations,
            "iteration_results": iteration_results,
            "final_strategy_weights": self.strategy_weights,
            "emergent_patterns": emergent_patterns,
            "chaos_evolution": self.chaos_history,
            "performance_evolution": self.performance_history,
        }

    def _analyze_emergence(self, iteration_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze emergent patterns from self-organization."""
        if not iteration_results:
            return {}

        # Extract time series
        chaos_series = [r["system_state"]["chaos_level"] for r in iteration_results]
        stability_series = [r["system_state"]["training_stability"] for r in iteration_results]

        # Compute trends
        chaos_trend = np.polyfit(range(len(chaos_series)), chaos_series, 1)[0]
        stability_trend = np.polyfit(range(len(stability_series)), stability_series, 1)[0]

        # Detect oscillations (simplified)
        chaos_fft = np.fft.fft(chaos_series)
        dominant_frequency = np.argmax(np.abs(chaos_fft[1:len(chaos_fft)//2])) + 1

        return {
            "chaos_trend": float(chaos_trend),
            "stability_trend": float(stability_trend),
            "oscillation_detected": dominant_frequency > 1,
            "dominant_period": int(len(chaos_series) / dominant_frequency) if dominant_frequency > 0 else None,
            "convergence": abs(chaos_trend) < 0.01 and abs(stability_trend) < 0.01,
        }


# ============================================================================
# Main Execution Examples
# ============================================================================

def main_example():
    """
    Demonstration of unified integration workflows.
    """
    # Initialize bridges
    kgirl = KGirlBridge(
        kgirl_url="http://localhost:9000",
        enable_chaos=True,
        enable_recursive=True,
    )

    orwells = OrwellsBridge(
        orwells_url="http://localhost:8000",
        enable_ml2=True,
    )

    # Example 1: Chaos-Aware Code Analysis
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Chaos-Aware Code Analysis")
    logger.info("="*60)

    chaos_pipeline = ChaosAwareCodeAnalysisPipeline(kgirl, orwells)

    # Generate dummy embeddings (in real use, these come from QVNM)
    dummy_embeddings = [
        np.random.randn(128).tolist() for _ in range(5)
    ]

    # Uncomment when you have a real CCL report:
    # chaos_results = chaos_pipeline.run(
    #     ccl_report_path="examples/report.min.json",
    #     query_embeddings=dummy_embeddings,
    # )
    # print(json.dumps(chaos_results, indent=2))

    # Example 2: Recursive Training Optimization
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Recursive Training Optimization")
    logger.info("="*60)

    training_pipeline = RecursiveTrainingOptimization(kgirl, orwells)

    initial_config = {
        "model": "tauls",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
    }

    # Uncomment to run:
    # training_results = training_pipeline.run(
    #     initial_config=initial_config,
    #     parameter_embeddings=dummy_embeddings,
    #     max_depth=3,
    # )
    # print(json.dumps(training_results, indent=2))

    # Example 3: Semantic Query to SQL
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Semantic Query to SQL")
    logger.info("="*60)

    sql_pipeline = SemanticQueryToSQL(kgirl, orwells)

    # Uncomment to run:
    # sql_results = sql_pipeline.run(
    #     natural_language_query="Find all functions with high entropy",
    #     vector_context=dummy_embeddings,
    #     schema_info={"tables": [{"name": "functions", "columns": ["id", "name", "entropy"]}]},
    # )
    # print(json.dumps(sql_results, indent=2))

    # Example 4: Self-Organizing Intelligence
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: Self-Organizing Intelligence")
    logger.info("="*60)

    soi = SelfOrganizingIntelligence(kgirl, orwells, learning_rate=0.2)

    # Uncomment to run:
    # soi_results = soi.self_organize(
    #     n_iterations=10,
    #     ccl_report_path="examples/report.min.json",
    #     vector_data=dummy_embeddings,
    # )
    # print(json.dumps(soi_results, indent=2))

    logger.info("\n" + "="*60)
    logger.info("All examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main_example()
