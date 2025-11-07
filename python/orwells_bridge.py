"""
orwells_bridge.py - Integration bridge for orwells-egg ecosystem

This module provides integration with the orwells-egg repository's infrastructure:
- RFV (Repository Function Vector): Versioned computational artifacts
- AA (Adaptive Agents): Priority queue and job scheduling
- DS (Data Selection): SQL generation and query optimization
- ML2 (Meta-Learning Layer 2): Neural architecture with custom training
- IA (Intelligent Automation): Workflow coordination

Integration Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │           SHOUT + ORWELLS-EGG INTEGRATION                │
    │                                                          │
    │  ┌──────────────┐           ┌──────────────────┐       │
    │  │ CCL Analysis │──────────▶│ RFV Snapshot     │       │
    │  │ (Functions)  │  vectorize │ (Code Artifacts) │       │
    │  └──────────────┘           └──────────────────┘       │
    │                                     │                    │
    │  ┌──────────────┐                  │                    │
    │  │ TA-ULS Train │◀─────────────────┘                    │
    │  │              │    RFV-guided                         │
    │  └──────┬───────┘    optimization                       │
    │         │                                                │
    │         ▼                                                │
    │  ┌──────────────┐           ┌──────────────────┐       │
    │  │ Job Queue    │──────────▶│ ML2 Training     │       │
    │  │ (AA)         │  priority  │ (Coach Policy)   │       │
    │  └──────────────┘           └──────────────────┘       │
    │                                                          │
    │  ┌──────────────┐           ┌──────────────────┐       │
    │  │ QVNM Query   │◀─────────▶│ SQL Generator    │       │
    │  │              │   semantic │ (DS)             │       │
    │  └──────────────┘           └──────────────────┘       │
    └──────────────────────────────────────────────────────────┘
"""

import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import json


class JobPriority(Enum):
    """Job priority levels for adaptive queue."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ArtifactType(Enum):
    """Types of computational artifacts for RFV system."""
    CODE = "code"
    MODEL = "model"
    DATA = "data"
    EMBEDDING = "embedding"
    ANALYSIS = "analysis"


@dataclass
class RFVSnapshot:
    """Repository Function Vector snapshot."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    artifact_type: ArtifactType = ArtifactType.CODE
    vector_embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_snapshot_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    version_tag: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for API calls."""
        return {
            "snapshot_id": self.snapshot_id,
            "artifact_type": self.artifact_type.value,
            "vector_embedding": self.vector_embedding,
            "metadata": self.metadata,
            "parent_snapshot_id": self.parent_snapshot_id,
            "created_at": self.created_at.isoformat(),
            "version_tag": self.version_tag,
        }


@dataclass
class JobDescriptor:
    """Job descriptor for adaptive priority queue."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: str = "generic"
    priority: JobPriority = JobPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    worker_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    leased_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "priority": self.priority.value,
            "payload": self.payload,
            "worker_id": self.worker_id,
            "created_at": self.created_at.isoformat(),
            "leased_at": self.leased_at.isoformat() if self.leased_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }


class OrwellsBridge:
    """
    Bridge to orwells-egg infrastructure.

    This class provides integration with:
    - RFV system for versioned computational artifacts
    - AA (Adaptive Agents) for job scheduling
    - DS (Data Selection) for SQL generation
    - ML2 for meta-learning and training
    - IA for workflow coordination
    """

    def __init__(
        self,
        orwells_url: str = "http://localhost:8000",
        postgres_connection: Optional[str] = None,
        enable_ml2: bool = True,
    ):
        """
        Initialize orwells-egg bridge.

        Args:
            orwells_url: Base URL for orwells-egg API
            postgres_connection: PostgreSQL connection string
            enable_ml2: Enable ML2 training integration
        """
        self.orwells_url = orwells_url
        self.postgres_connection = postgres_connection
        self.enable_ml2 = enable_ml2
        self.pg_conn = None

        if postgres_connection:
            self._init_postgres()

    def _init_postgres(self):
        """Initialize PostgreSQL connection."""
        try:
            import psycopg2
            self.pg_conn = psycopg2.connect(self.postgres_connection)
            print("PostgreSQL connection established for orwells-egg")
        except ImportError:
            print("Warning: psycopg2 not installed")
        except Exception as e:
            print(f"Warning: Could not connect to PostgreSQL: {e}")

    # ========================================================================
    # RFV (Repository Function Vector) Operations
    # ========================================================================

    def create_rfv_snapshot(
        self,
        artifact_type: ArtifactType,
        vector_embedding: List[float],
        metadata: Dict[str, Any],
        parent_snapshot_id: Optional[str] = None,
        version_tag: Optional[str] = None,
    ) -> RFVSnapshot:
        """
        Create RFV snapshot for computational artifact.

        Args:
            artifact_type: Type of artifact (code, model, data, etc.)
            vector_embedding: Vector representation of artifact
            metadata: Associated metadata
            parent_snapshot_id: Parent snapshot for version history
            version_tag: Optional version tag (e.g., "v1.0.0")

        Returns:
            Created RFV snapshot
        """
        snapshot = RFVSnapshot(
            artifact_type=artifact_type,
            vector_embedding=vector_embedding,
            metadata=metadata,
            parent_snapshot_id=parent_snapshot_id,
            version_tag=version_tag,
        )

        try:
            response = requests.post(
                f"{self.orwells_url}/rfv/publish",
                json=snapshot.to_dict(),
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            snapshot.snapshot_id = result.get("snapshot_id", snapshot.snapshot_id)
        except requests.exceptions.RequestException as e:
            print(f"RFV publish failed, using local snapshot: {e}")

        return snapshot

    def query_rfv_history(
        self,
        snapshot_id: str,
        max_depth: int = 10,
    ) -> List[RFVSnapshot]:
        """
        Query RFV version history.

        Args:
            snapshot_id: Starting snapshot ID
            max_depth: Maximum history depth

        Returns:
            List of snapshots in version history
        """
        try:
            response = requests.get(
                f"{self.orwells_url}/rfv/history/{snapshot_id}",
                params={"max_depth": max_depth},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            snapshots = []
            for snap_data in data.get("history", []):
                snapshot = RFVSnapshot(
                    snapshot_id=snap_data["snapshot_id"],
                    artifact_type=ArtifactType(snap_data["artifact_type"]),
                    vector_embedding=snap_data["vector_embedding"],
                    metadata=snap_data["metadata"],
                    parent_snapshot_id=snap_data.get("parent_snapshot_id"),
                    version_tag=snap_data.get("version_tag"),
                )
                snapshots.append(snapshot)

            return snapshots
        except requests.exceptions.RequestException as e:
            print(f"RFV history query failed: {e}")
            return []

    def rfv_semantic_diff(
        self,
        snapshot_id_a: str,
        snapshot_id_b: str,
    ) -> Dict[str, Any]:
        """
        Compute semantic diff between two RFV snapshots.

        Args:
            snapshot_id_a: First snapshot ID
            snapshot_id_b: Second snapshot ID

        Returns:
            Semantic difference analysis
        """
        try:
            response = requests.post(
                f"{self.orwells_url}/rfv/diff",
                json={
                    "snapshot_a": snapshot_id_a,
                    "snapshot_b": snapshot_id_b,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"RFV diff failed: {e}")
            return {"error": str(e)}

    # ========================================================================
    # AA (Adaptive Agents) - Priority Queue Operations
    # ========================================================================

    def submit_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
    ) -> JobDescriptor:
        """
        Submit job to adaptive priority queue.

        Args:
            job_type: Type of job (e.g., "training", "analysis", "optimization")
            payload: Job payload data
            priority: Job priority level

        Returns:
            Job descriptor
        """
        job = JobDescriptor(
            job_type=job_type,
            payload=payload,
            priority=priority,
        )

        try:
            response = requests.post(
                f"{self.orwells_url}/pq/submit",
                json=job.to_dict(),
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            job.job_id = result.get("job_id", job.job_id)
            print(f"Job submitted: {job.job_id} (priority: {priority.name})")
        except requests.exceptions.RequestException as e:
            print(f"Job submission failed: {e}")

        return job

    def lease_job(self, worker_id: str) -> Optional[JobDescriptor]:
        """
        Lease next job from priority queue.

        Args:
            worker_id: Worker identifier

        Returns:
            Leased job descriptor or None if queue empty
        """
        try:
            response = requests.post(
                f"{self.orwells_url}/pq/lease",
                json={"worker_id": worker_id},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "job_id" not in data:
                return None

            job = JobDescriptor(
                job_id=data["job_id"],
                job_type=data["job_type"],
                priority=JobPriority(data["priority"]),
                payload=data["payload"],
                worker_id=worker_id,
                leased_at=datetime.now(),
            )
            print(f"Job leased: {job.job_id} by worker {worker_id}")
            return job
        except requests.exceptions.RequestException as e:
            print(f"Job lease failed: {e}")
            return None

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> bool:
        """
        Mark job as completed.

        Args:
            job_id: Job identifier
            result: Job result data

        Returns:
            Success status
        """
        try:
            response = requests.post(
                f"{self.orwells_url}/pq/complete",
                json={"job_id": job_id, "result": result},
                timeout=30,
            )
            response.raise_for_status()
            print(f"Job completed: {job_id}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Job completion failed: {e}")
            return False

    def adapt_priority(
        self,
        job_id: str,
        new_priority: JobPriority,
        reason: str,
    ) -> bool:
        """
        Adapt job priority dynamically.

        Args:
            job_id: Job identifier
            new_priority: New priority level
            reason: Reason for priority change

        Returns:
            Success status
        """
        try:
            response = requests.post(
                f"{self.orwells_url}/pq/adapt",
                json={
                    "job_id": job_id,
                    "new_priority": new_priority.value,
                    "reason": reason,
                },
                timeout=30,
            )
            response.raise_for_status()
            print(f"Job priority adapted: {job_id} -> {new_priority.name}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Priority adaptation failed: {e}")
            return False

    # ========================================================================
    # DS (Data Selection) - SQL Generation
    # ========================================================================

    def generate_sql(
        self,
        natural_language_query: str,
        schema_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate SQL from natural language query.

        Args:
            natural_language_query: Natural language description
            schema_context: Optional database schema context

        Returns:
            Generated SQL and metadata
        """
        try:
            response = requests.post(
                f"{self.orwells_url}/ds/generate_sql",
                json={
                    "query": natural_language_query,
                    "schema": schema_context,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"SQL generation failed: {e}")
            return {"error": str(e)}

    def execute_generated_sql(
        self,
        sql_query: str,
        log_execution: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute generated SQL with logging.

        Args:
            sql_query: SQL query to execute
            log_execution: Whether to log execution

        Returns:
            Query results
        """
        if not self.pg_conn:
            return {"error": "PostgreSQL connection not initialized"}

        try:
            cursor = self.pg_conn.cursor()
            cursor.execute(sql_query)

            # Fetch results
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
            else:
                results = []

            self.pg_conn.commit()

            # Log execution if enabled
            if log_execution:
                self._log_sql_execution(sql_query, len(results))

            return {
                "results": results,
                "row_count": len(results),
            }
        except Exception as e:
            self.pg_conn.rollback()
            return {"error": str(e)}

    def _log_sql_execution(self, sql_query: str, row_count: int):
        """Log SQL execution to orwells-egg."""
        try:
            requests.post(
                f"{self.orwells_url}/ds/log_query",
                json={
                    "query": sql_query,
                    "row_count": row_count,
                    "timestamp": datetime.now().isoformat(),
                },
                timeout=5,
            )
        except Exception:
            pass  # Silent failure for logging

    # ========================================================================
    # ML2 (Meta-Learning Layer 2) - Training Integration
    # ========================================================================

    def ml2_training_step(
        self,
        model_state: Dict[str, Any],
        batch_data: List[List[float]],
        coach_policy: str = "entropy_aware",
    ) -> Dict[str, Any]:
        """
        Execute ML2 training step with coach policy.

        Args:
            model_state: Current model state
            batch_data: Training batch
            coach_policy: Coach policy strategy

        Returns:
            Training step results
        """
        if not self.enable_ml2:
            return {"error": "ML2 integration disabled"}

        try:
            response = requests.post(
                f"{self.orwells_url}/ml2/train_step",
                json={
                    "model_state": model_state,
                    "batch_data": batch_data,
                    "coach_policy": coach_policy,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"ML2 training step failed: {e}")
            return {"error": str(e)}

    def ml2_update_coach(
        self,
        metrics: Dict[str, float],
        gradient_stats: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Update ML2 coach state based on training metrics.

        Args:
            metrics: Training metrics (loss, accuracy, etc.)
            gradient_stats: Gradient statistics

        Returns:
            Updated coach recommendations
        """
        if not self.enable_ml2:
            return {"error": "ML2 integration disabled"}

        try:
            response = requests.post(
                f"{self.orwells_url}/ml2/update_coach",
                json={
                    "metrics": metrics,
                    "gradient_stats": gradient_stats,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"ML2 coach update failed: {e}")
            return {"error": str(e)}

    # ========================================================================
    # IA (Intelligent Automation) - Workflow Coordination
    # ========================================================================

    def create_workflow(
        self,
        workflow_name: str,
        steps: List[Dict[str, Any]],
    ) -> str:
        """
        Create coordinated workflow.

        Args:
            workflow_name: Name of workflow
            steps: List of workflow steps

        Returns:
            Workflow ID
        """
        workflow_id = str(uuid.uuid4())

        try:
            response = requests.post(
                f"{self.orwells_url}/ia/workflow/create",
                json={
                    "workflow_id": workflow_id,
                    "name": workflow_name,
                    "steps": steps,
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("workflow_id", workflow_id)
        except requests.exceptions.RequestException as e:
            print(f"Workflow creation failed: {e}")
            return workflow_id

    def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute coordinated workflow.

        Args:
            workflow_id: Workflow identifier
            input_data: Input data for workflow

        Returns:
            Workflow execution results
        """
        try:
            response = requests.post(
                f"{self.orwells_url}/ia/workflow/execute",
                json={
                    "workflow_id": workflow_id,
                    "input_data": input_data,
                },
                timeout=300,  # Long timeout for workflow execution
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Workflow execution failed: {e}")
            return {"error": str(e)}


# ============================================================================
# Integration Use Cases
# ============================================================================

def integrate_ccl_with_rfv(
    ccl_report_path: str,
    orwells_bridge: OrwellsBridge,
    vector_embeddings: List[List[float]],
) -> Dict[str, Any]:
    """
    Integration use case: Create RFV snapshots from CCL analysis.

    Args:
        ccl_report_path: Path to CCL JSON report
        orwells_bridge: Initialized OrwellsBridge instance
        vector_embeddings: Vector embeddings for functions

    Returns:
        Created RFV snapshots
    """
    # Load CCL report
    with open(ccl_report_path, 'r') as f:
        ccl_report = json.load(f)

    functions = ccl_report.get("functions", [])
    snapshots = []

    # Create RFV snapshot for each function
    for i, func in enumerate(functions):
        if i >= len(vector_embeddings):
            break

        metadata = {
            "function_name": func.get("name", f"func_{i}"),
            "entropy_bits": func.get("entropy_bits", 0),
            "ghost_score": func.get("ghost_score", 0),
            "purity": func.get("purity", "unknown"),
            "file_path": func.get("file", "unknown"),
        }

        snapshot = orwells_bridge.create_rfv_snapshot(
            artifact_type=ArtifactType.CODE,
            vector_embedding=vector_embeddings[i],
            metadata=metadata,
            version_tag=ccl_report.get("version", "v1.0.0"),
        )
        snapshots.append(snapshot)

    return {
        "n_snapshots": len(snapshots),
        "snapshots": [s.to_dict() for s in snapshots],
    }


def integrate_tauls_with_job_queue(
    orwells_bridge: OrwellsBridge,
    training_configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Integration use case: Submit TA-ULS training jobs to adaptive queue.

    Args:
        orwells_bridge: Initialized OrwellsBridge instance
        training_configs: List of training configurations

    Returns:
        Submitted job descriptors
    """
    jobs = []

    for config in training_configs:
        # Determine priority based on training urgency
        priority = JobPriority.HIGH if config.get("urgent", False) else JobPriority.NORMAL

        job = orwells_bridge.submit_job(
            job_type="tauls_training",
            payload=config,
            priority=priority,
        )
        jobs.append(job)

    return {
        "n_jobs": len(jobs),
        "jobs": [j.to_dict() for j in jobs],
    }


def integrate_qvnm_with_sql_generation(
    orwells_bridge: OrwellsBridge,
    query_text: str,
    qvnm_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Integration use case: Use QVNM semantic understanding for SQL generation.

    Args:
        orwells_bridge: Initialized OrwellsBridge instance
        query_text: Natural language query
        qvnm_context: QVNM semantic context

    Returns:
        Generated SQL with semantic grounding
    """
    # Enhance query with QVNM semantic context
    enhanced_query = f"{query_text}\n\nSemantic context: {json.dumps(qvnm_context)}"

    # Generate SQL
    sql_result = orwells_bridge.generate_sql(
        natural_language_query=enhanced_query,
        schema_context=qvnm_context.get("schema", None),
    )

    return sql_result
