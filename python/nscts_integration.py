"""
NSCTS Integration with kgirl and orwells-egg ecosystems

This module demonstrates advanced integration possibilities:
1. Consciousness state vectors → QVNM semantic analysis
2. Coherence metrics → ChaosRAG adaptive retrieval
3. Training sessions → RFV versioning
4. Biometric patterns → Holographic memory storage
5. Session scheduling → Adaptive job queue

Architecture:
    ┌────────────────────────────────────────────────────────┐
    │         NSCTS + kgirl + orwells-egg Integration        │
    │                                                        │
    │  ┌──────────────┐    ┌──────────────┐   ┌──────────┐ │
    │  │   NSCTS      │───▶│   kgirl      │──▶│ orwells  │ │
    │  │              │    │              │   │          │ │
    │  │ Biometrics   │    │ Chaos from   │   │ RFV      │ │
    │  │ Coherence    │    │ coherence    │   │ Version  │ │
    │  │ States       │    │              │   │ Sessions │ │
    │  │              │    │ Holographic  │   │          │ │
    │  │ Trinity      │    │ Memory for   │   │ Job      │ │
    │  │ Flow         │    │ Patterns     │   │ Queue    │ │
    │  └──────────────┘    └──────────────┘   └──────────┘ │
    └────────────────────────────────────────────────────────┘
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from nscts import (
    ConsciousnessTrainingSystem,
    ConsciousnessState,
    BiometricSignature,
    CoherenceState,
    LearningPhase,
)
from kgirl_bridge import KGirlBridge, ChaosMode
from orwells_bridge import OrwellsBridge, ArtifactType, JobPriority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Consciousness → Vector Embedding
# ============================================================================

def consciousness_to_embedding(state: ConsciousnessState) -> List[float]:
    """
    Convert consciousness state to high-dimensional vector for QVNM/kgirl integration.

    Returns 17-dimensional vector encoding all biometric and coherence features.
    """
    return state.to_vector()


def compute_consciousness_chaos(states: List[ConsciousnessState]) -> float:
    """
    Compute chaos metric from consciousness state history.

    Uses coherence variability and state transitions as chaos indicators.

    Args:
        states: Recent consciousness states

    Returns:
        Chaos metric in [0, 1]
    """
    if not states or len(states) < 2:
        return 0.5

    # Extract coherence time series
    trinity_series = np.array([s.trinity_coherence for s in states])
    global_series = np.array([s.global_coherence for s in states])

    # Variability indicates chaos
    trinity_var = np.std(trinity_series)
    global_var = np.std(global_series)

    # State transitions indicate chaos
    state_changes = sum(
        1 for i in range(1, len(states))
        if states[i].detect_state() != states[i-1].detect_state()
    )
    transition_rate = state_changes / len(states)

    # Flow disruption indicates chaos
    flow_disruptions = sum(1 for s in states if s.flow_state < 0.5)
    disruption_rate = flow_disruptions / len(states)

    # Combined chaos metric
    chaos = (
        0.3 * trinity_var * 10 +           # Trinity variability (scaled)
        0.3 * global_var * 10 +            # Global variability (scaled)
        0.2 * transition_rate +            # State transition frequency
        0.2 * disruption_rate              # Flow disruptions
    )

    return float(np.clip(chaos, 0, 1))


# ============================================================================
# Integration Pipeline 1: Consciousness-Aware Retrieval
# ============================================================================

class ConsciousnessAwareRetrieval:
    """
    Use consciousness states to drive chaos-adaptive retrieval.

    When coherence is high (low chaos): Precise, focused retrieval
    When coherence is low (high chaos): Diverse, exploratory retrieval
    """

    def __init__(self, kgirl_bridge: KGirlBridge):
        self.kgirl = kgirl_bridge
        self.consciousness_history: List[ConsciousnessState] = []

    def update(self, state: ConsciousnessState):
        """Update with latest consciousness state."""
        self.consciousness_history.append(state)
        if len(self.consciousness_history) > 100:
            self.consciousness_history.pop(0)

    def query(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute query with consciousness-driven chaos adaptation.

        Args:
            query_text: Natural language query
            query_embedding: Query vector embedding
            top_k: Number of results

        Returns:
            Retrieval results with consciousness context
        """
        if not self.consciousness_history:
            # No consciousness data - use moderate chaos
            chaos_mode = ChaosMode.MODERATE
            chaos_metric = 0.5
        else:
            # Compute chaos from recent consciousness states
            chaos_metric = compute_consciousness_chaos(self.consciousness_history[-20:])
            chaos_mode = self.kgirl.determine_chaos_mode(chaos_metric)

        latest_state = self.consciousness_history[-1] if self.consciousness_history else None

        logger.info(
            f"Consciousness-aware query: chaos={chaos_metric:.3f}, mode={chaos_mode.value}, "
            f"coherence={latest_state.global_coherence:.3f if latest_state else 'N/A'}"
        )

        # Execute chaos-adaptive retrieval
        results = self.kgirl.chaos_rag_query(
            query_vector=query_embedding,
            chaos_mode=chaos_mode,
            top_k=top_k,
            time_frequency_analysis=True,
        )

        # Enrich with consciousness context
        results['consciousness_context'] = {
            'chaos_metric': chaos_metric,
            'chaos_mode': chaos_mode.value,
            'current_state': {
                'global_coherence': latest_state.global_coherence,
                'trinity_coherence': latest_state.trinity_coherence,
                'flow_state': latest_state.flow_state,
                'presence': latest_state.presence,
                'love_field': latest_state.love_field,
                'coherence_state': latest_state.detect_state().value,
            } if latest_state else None,
        }

        return results


# ============================================================================
# Integration Pipeline 2: Consciousness Pattern Memory
# ============================================================================

class ConsciousnessPatternMemory:
    """
    Store consciousness patterns in holographic memory for associative recall.

    High-coherence states become attractors that can be recalled and replayed.
    """

    def __init__(self, kgirl_bridge: KGirlBridge):
        self.kgirl = kgirl_bridge
        self.pattern_library: List[Dict[str, Any]] = []

    def store_high_coherence_state(self, state: ConsciousnessState, threshold: float = 0.75):
        """
        Store consciousness state in holographic memory if coherence exceeds threshold.

        Args:
            state: Consciousness state to potentially store
            threshold: Minimum global coherence for storage
        """
        if state.global_coherence < threshold:
            return

        # Convert to vector embedding
        pattern_vector = consciousness_to_embedding(state)

        # Store in holographic memory
        memory_result = self.kgirl.holographic_memory_recall(
            pattern=pattern_vector,
            association_depth=2,
        )

        # Track in local library
        self.pattern_library.append({
            'timestamp': state.timestamp,
            'vector': pattern_vector,
            'coherence': state.global_coherence,
            'state_type': state.detect_state().value,
            'trinity': state.trinity_coherence,
            'flow': state.flow_state,
            'presence': state.presence,
            'love': state.love_field,
            'memory_result': memory_result,
        })

        logger.info(
            f"Stored high-coherence pattern: coherence={state.global_coherence:.3f}, "
            f"state={state.detect_state().value}"
        )

    def recall_similar_states(
        self,
        current_state: ConsciousnessState,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recall similar consciousness patterns from memory.

        Args:
            current_state: Current consciousness state
            top_k: Number of similar patterns to recall

        Returns:
            List of similar patterns from library
        """
        if not self.pattern_library:
            return []

        current_vec = np.array(consciousness_to_embedding(current_state))

        # Compute similarities
        similarities = []
        for pattern in self.pattern_library:
            pattern_vec = np.array(pattern['vector'])
            # Cosine similarity
            sim = np.dot(current_vec, pattern_vec) / (
                np.linalg.norm(current_vec) * np.linalg.norm(pattern_vec) + 1e-12
            )
            similarities.append((sim, pattern))

        # Sort by similarity and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [
            {**pattern, 'similarity': float(sim)}
            for sim, pattern in similarities[:top_k]
        ]


# ============================================================================
# Integration Pipeline 3: Session Versioning with RFV
# ============================================================================

class ConsciousnessSessionVersioning:
    """
    Version consciousness training sessions using RFV system.

    Each session becomes a versioned artifact with:
    - Session metrics as metadata
    - Consciousness trajectory as vector embedding
    - Parent-child relationships for session evolution
    """

    def __init__(self, orwells_bridge: OrwellsBridge):
        self.orwells = orwells_bridge
        self.session_snapshots: Dict[str, str] = {}  # human_id -> snapshot_id

    def version_session(
        self,
        human_id: str,
        session_report: Dict[str, Any],
        consciousness_trajectory: List[ConsciousnessState],
    ) -> str:
        """
        Create RFV snapshot for training session.

        Args:
            human_id: Human identifier
            session_report: Session report from NSCTS trainer
            consciousness_trajectory: Consciousness states from session

        Returns:
            RFV snapshot ID
        """
        # Compute trajectory embedding (average of all state vectors)
        if consciousness_trajectory:
            vectors = np.array([consciousness_to_embedding(s) for s in consciousness_trajectory])
            trajectory_embedding = np.mean(vectors, axis=0).tolist()
        else:
            trajectory_embedding = [0.0] * 17

        # Get parent snapshot if exists
        parent_id = self.session_snapshots.get(human_id)

        # Create metadata
        metadata = {
            'human_id': human_id,
            'session_number': session_report['session']['sessions'],
            'learning_phase': session_report['session']['phase'],
            'metrics': session_report['metrics'],
            'state_distribution': session_report['distribution'],
            'sync_quality': session_report['sync_quality'],
            'n_events': session_report['session']['events'],
            'trajectory_length': len(consciousness_trajectory),
            'timestamp': datetime.now().isoformat(),
        }

        # Create RFV snapshot
        snapshot = self.orwells.create_rfv_snapshot(
            artifact_type=ArtifactType.ANALYSIS,
            vector_embedding=trajectory_embedding,
            metadata=metadata,
            parent_snapshot_id=parent_id,
            version_tag=f"session_{session_report['session']['sessions']}",
        )

        # Update tracking
        self.session_snapshots[human_id] = snapshot.snapshot_id

        logger.info(
            f"Versioned session {session_report['session']['sessions']} for {human_id}: "
            f"snapshot_id={snapshot.snapshot_id}"
        )

        return snapshot.snapshot_id

    def get_session_evolution(self, human_id: str, max_depth: int = 20) -> List[Dict[str, Any]]:
        """
        Query session evolution history via RFV.

        Args:
            human_id: Human identifier
            max_depth: Maximum history depth

        Returns:
            Session evolution history
        """
        if human_id not in self.session_snapshots:
            return []

        snapshot_id = self.session_snapshots[human_id]
        history = self.orwells.query_rfv_history(snapshot_id, max_depth)

        return [
            {
                'snapshot_id': snap.snapshot_id,
                'session_number': snap.metadata.get('session_number'),
                'learning_phase': snap.metadata.get('learning_phase'),
                'metrics': snap.metadata.get('metrics'),
                'timestamp': snap.metadata.get('timestamp'),
            }
            for snap in history
        ]


# ============================================================================
# Integration Pipeline 4: Distributed Consciousness Training
# ============================================================================

class DistributedConsciousnessTraining:
    """
    Distribute consciousness training sessions via adaptive job queue.

    Multiple humans can train simultaneously with coordinated scheduling.
    """

    def __init__(self, orwells_bridge: OrwellsBridge):
        self.orwells = orwells_bridge
        self.active_sessions: Dict[str, str] = {}  # human_id -> job_id

    def schedule_session(
        self,
        human_id: str,
        session_config: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
    ) -> str:
        """
        Schedule consciousness training session via job queue.

        Args:
            human_id: Human identifier
            session_config: Session configuration
            priority: Job priority

        Returns:
            Job ID
        """
        # Prepare job payload
        payload = {
            'human_id': human_id,
            'session_config': session_config,
            'timestamp': datetime.now().isoformat(),
        }

        # Submit to job queue
        job = self.orwells.submit_job(
            job_type='consciousness_training',
            payload=payload,
            priority=priority,
        )

        self.active_sessions[human_id] = job.job_id
        logger.info(f"Scheduled session for {human_id}: job_id={job.job_id}")

        return job.job_id

    def complete_session(
        self,
        human_id: str,
        session_report: Dict[str, Any],
    ) -> bool:
        """
        Mark session as completed in job queue.

        Args:
            human_id: Human identifier
            session_report: Session results

        Returns:
            Success status
        """
        if human_id not in self.active_sessions:
            logger.warning(f"No active session for {human_id}")
            return False

        job_id = self.active_sessions[human_id]
        success = self.orwells.complete_job(job_id, session_report)

        if success:
            del self.active_sessions[human_id]
            logger.info(f"Completed session for {human_id}")

        return success


# ============================================================================
# Unified Integration Example
# ============================================================================

async def unified_consciousness_workflow(
    human_id: str,
    data_source,
    kgirl: KGirlBridge,
    orwells: OrwellsBridge,
    session_minutes: float = 2.0,
) -> Dict[str, Any]:
    """
    Complete workflow integrating NSCTS, kgirl, and orwells-egg.

    Workflow:
    1. Run consciousness training session (NSCTS)
    2. Store high-coherence patterns in holographic memory (kgirl)
    3. Version session with RFV (orwells-egg)
    4. Perform consciousness-aware queries (kgirl)
    5. Return comprehensive results

    Args:
        human_id: Human identifier
        data_source: Biometric data source
        kgirl: kgirl bridge
        orwells: orwells-egg bridge
        session_minutes: Session duration

    Returns:
        Unified workflow results
    """
    logger.info(f"Starting unified consciousness workflow for {human_id}")

    # Initialize integration components
    retrieval = ConsciousnessAwareRetrieval(kgirl)
    memory = ConsciousnessPatternMemory(kgirl)
    versioning = ConsciousnessSessionVersioning(orwells)

    # Step 1: Run NSCTS training session
    logger.info("Step 1: Running NSCTS training session...")
    nscts = ConsciousnessTrainingSystem()
    session_report = await nscts.train(human_id, data_source, session_minutes)

    # Get consciousness trajectory
    trainer = nscts.get(human_id)
    trajectory = trainer.iface.history[-100:]  # Last 100 states

    # Step 2: Store high-coherence patterns
    logger.info("Step 2: Storing high-coherence patterns in holographic memory...")
    for state in trajectory:
        retrieval.update(state)
        memory.store_high_coherence_state(state, threshold=0.70)

    logger.info(f"Stored {len(memory.pattern_library)} high-coherence patterns")

    # Step 3: Version session with RFV
    logger.info("Step 3: Versioning session with RFV...")
    snapshot_id = versioning.version_session(human_id, session_report, trajectory)

    # Step 4: Perform consciousness-aware query
    logger.info("Step 4: Performing consciousness-aware query...")
    latest_state = trajectory[-1] if trajectory else None
    if latest_state:
        query_embedding = consciousness_to_embedding(latest_state)
        query_results = retrieval.query(
            query_text=f"Find similar consciousness patterns for {human_id}",
            query_embedding=query_embedding,
            top_k=5,
        )
    else:
        query_results = {}

    # Step 5: Recall similar patterns
    logger.info("Step 5: Recalling similar consciousness patterns...")
    if latest_state:
        similar_patterns = memory.recall_similar_states(latest_state, top_k=5)
    else:
        similar_patterns = []

    # Step 6: Get session evolution
    logger.info("Step 6: Retrieving session evolution history...")
    evolution = versioning.get_session_evolution(human_id, max_depth=10)

    # Compile results
    results = {
        'human_id': human_id,
        'session_report': session_report,
        'rfv_snapshot_id': snapshot_id,
        'trajectory_stats': {
            'n_states': len(trajectory),
            'avg_coherence': float(np.mean([s.global_coherence for s in trajectory])) if trajectory else 0,
            'avg_flow': float(np.mean([s.flow_state for s in trajectory])) if trajectory else 0,
            'avg_presence': float(np.mean([s.presence for s in trajectory])) if trajectory else 0,
            'avg_love': float(np.mean([s.love_field for s in trajectory])) if trajectory else 0,
        },
        'consciousness_aware_query': query_results,
        'similar_patterns': similar_patterns,
        'session_evolution': evolution,
        'pattern_library_size': len(memory.pattern_library),
    }

    logger.info(f"Unified workflow complete for {human_id}")
    return results


# ============================================================================
# Demo
# ============================================================================

async def demo_integrated_consciousness():
    """Demonstrate unified integration."""
    from nscts import demo_data_source

    print("\n" + "="*80)
    print("INTEGRATED CONSCIOUSNESS TRAINING DEMO")
    print("NSCTS + kgirl + orwells-egg")
    print("="*80 + "\n")

    # Initialize bridges
    kgirl = KGirlBridge(kgirl_url="http://localhost:9000", enable_chaos=True)
    orwells = OrwellsBridge(orwells_url="http://localhost:8000", enable_ml2=True)

    # Run unified workflow
    results = await unified_consciousness_workflow(
        human_id="integration_demo",
        data_source=demo_data_source,
        kgirl=kgirl,
        orwells=orwells,
        session_minutes=0.5,  # 30 seconds
    )

    # Print results
    print("\nUNIFIED WORKFLOW RESULTS")
    print("-"*50)
    print(f"Human ID: {results['human_id']}")
    print(f"RFV Snapshot: {results['rfv_snapshot_id']}")
    print(f"\nTrajectory Statistics:")
    for k, v in results['trajectory_stats'].items():
        print(f"  {k:20}: {v:.3f}" if isinstance(v, float) else f"  {k:20}: {v}")
    print(f"\nPattern Library Size: {results['pattern_library_size']}")
    print(f"Similar Patterns Found: {len(results['similar_patterns'])}")
    print(f"Session Evolution History: {len(results['session_evolution'])} sessions")

    if results['consciousness_aware_query'] and 'consciousness_context' in results['consciousness_aware_query']:
        ctx = results['consciousness_aware_query']['consciousness_context']
        print(f"\nConsciousness-Aware Query Context:")
        print(f"  Chaos Metric: {ctx['chaos_metric']:.3f}")
        print(f"  Chaos Mode: {ctx['chaos_mode']}")
        if ctx['current_state']:
            print(f"  Current Coherence: {ctx['current_state']['global_coherence']:.3f}")
            print(f"  Current State: {ctx['current_state']['coherence_state']}")

    print("\n" + "="*80)
    print("Integration demo complete!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(demo_integrated_consciousness())
