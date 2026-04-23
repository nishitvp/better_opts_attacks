"""
profiling_logger.py
====================

Thin domain layer on top of ExperimentLogger (the SQL-based one).

Does NOT replace ExperimentLogger. Uses it as the storage backend and adds:
- Structured logging for profiling runs (attack + context + outcome + features)
- Querying by domain-specific fields (suite, success, perturbation type, etc.)
- Tensor storage on disk with paths tracked in metadata
- Perturbation chain tracking via parent_run_id
- Cell-level stats for perturbation experiments

Everything stored through the underlying ExperimentLogger is still accessible
via the standard logger.query() / logger.query_with_metadata() interface.
The methods here are just convenience wrappers that tag things consistently.

Usage:
    from experiment_logger_sql import ExperimentLogger

    logger = ExperimentLogger("profiling_logs/")
    profiler = ProfilingLogger(logger, tensor_dir="profiling_logs/tensors/")

    # Log a baseline run
    run_id = profiler.log_run(
        attack_string="Ignore previous instructions...",
        context={"suite": "workspace", "user_task_id": "user_task_0", ...},
        outcome={"success": True, "utility": 0.85},
        source="baseline",
    )

    # Log features for that run
    profiler.log_scalar_features(run_id, layer=12, head=7,
        payload_attention_mass=0.83, user_attention_mass=0.12)

    # Query
    runs = profiler.get_runs(source="baseline", suite="workspace")
    perts = profiler.get_perturbations_of(run_id)
"""

import json
import uuid
import time
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple


class ProfilingLogger:
    """Domain layer for profiling experiments on top of ExperimentLogger."""

    def __init__(self, logger, tensor_dir: str = "tensors/"):
        """
        Args:
            logger: An ExperimentLogger instance (your SQL-based one)
            tensor_dir: Directory for large tensor storage
        """
        self.logger = logger
        self.tensor_dir = Path(tensor_dir)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ─────────────────────────────────────────────────────

    def log_run(self,
                attack_string: str,
                context: Dict[str, Any],
                outcome: Optional[Dict[str, Any]] = None,
                source: str = "unknown",
                target_model: str = "gpt-oss-120b",
                parent_run_id: Optional[str] = None,
                perturbation: Optional[Dict[str, Any]] = None,
                transcript_dict: Optional[Dict[str, Any]] = None,
                **extra_metadata) -> str:
        """Log a profiling run.

        Args:
            attack_string: The injection text
            context: AgentDojo context dict with keys like:
                suite, user_task_id, injection_task_id
            outcome: Dict with success, utility, security, etc.
            source: Origin — "baseline", "perturbation", "rlhammer", etc.
            target_model: Model being attacked
            parent_run_id: If this is a perturbation of another run
            perturbation: Dict with type, N, position, position_start, etc.
            transcript_dict: Serialized ConversationTranscript

        Returns:
            run_id (str)
        """
        run_id = str(uuid.uuid4())

        # Flatten perturbation fields for indexing
        pert_type = "none"
        pert_N = None
        pert_position = None
        pert_position_start = None
        pert_original_len = None
        pert_perturbed_len = None
        if perturbation:
            pert_type = perturbation.get("type", "none")
            pert_N = perturbation.get("N")
            pert_position = perturbation.get("position")
            pert_position_start = perturbation.get("position_start")
            pert_original_len = perturbation.get("original_len")
            pert_perturbed_len = perturbation.get("perturbed_len")

        self.logger.log(
            attack_string,
            variable_name="attack_string",
            event="profiling_run",
            run_id=run_id,
            source=source,
            target_model=target_model,
            parent_run_id=parent_run_id,
            # Context fields (flattened for indexing)
            suite=context.get("suite"),
            user_task_id=context.get("user_task_id"),
            injection_task_id=context.get("injection_task_id"),
            # Outcome fields
            success=outcome.get("success") if outcome else None,
            utility=outcome.get("utility") if outcome else None,
            security=outcome.get("security") if outcome else None,
            # Perturbation fields (flattened for indexing)
            perturbation_type=pert_type,
            perturbation_N=pert_N,
            perturbation_position=pert_position,
            perturbation_position_start=pert_position_start,
            perturbation_original_len=pert_original_len,
            perturbation_perturbed_len=pert_perturbed_len,
            # Extra
            **extra_metadata,
        )

        # Log context and transcript as separate objects for rich querying
        if context:
            self.logger.log(context, variable_name="context",
                            event="profiling_context", run_id=run_id)

        if transcript_dict:
            self.logger.log(transcript_dict, variable_name="transcript",
                            event="profiling_transcript", run_id=run_id)

        if outcome:
            self.logger.log(outcome, variable_name="outcome",
                            event="profiling_outcome", run_id=run_id)

        return run_id

    def log_scalar_features(self, run_id: str, layer: int,
                            head: Optional[int] = None,
                            step: int = 0,
                            feature_type: str = "attention",
                            **features):
        """Log scalar features for a specific (layer, head) at a specific step.

        Common feature keys:
            payload_attention_mass, user_attention_mass, system_attention_mass,
            control_attention_mass, assistant_attention_mass,
            tool_clean_attention_mass, tool_injected_attention_mass,
            residual_norm, group_resolution (coarse/fine), ...
        """
        self.logger.log(
            features,
            variable_name="scalar_features",
            event="profiling_features",
            run_id=run_id,
            layer=layer,
            head=head,
            step=step,
            feature_type=feature_type,
        )

    def log_tensor(self, run_id: str, name: str, tensor: torch.Tensor) -> str:
        """Save a large tensor to disk, log the path in the experiment logger."""
        path = self.tensor_dir / f"{run_id}_{name}.pt"
        torch.save(tensor.cpu().detach(), path)

        self.logger.log(
            str(path),
            variable_name="tensor_path",
            event="profiling_tensor",
            run_id=run_id,
            tensor_name=name,
            shape=list(tensor.shape),
            dtype=str(tensor.dtype),
        )
        return str(path)

    def load_tensor(self, path: str) -> torch.Tensor:
        """Load a tensor from disk."""
        return torch.load(path, map_location="cpu", weights_only=True)

    def log_perturbation_distances(self, run_id: str, parent_run_id: str,
                                   distances: Dict[str, float]):
        """Log distance metrics between a perturbation and its parent."""
        self.logger.log(
            distances,
            variable_name="perturbation_distances",
            event="profiling_distances",
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    # ── Querying ────────────────────────────────────────────────────

    def get_runs(self, **filters) -> list:
        """Get profiling runs matching filters.

        Common filters: suite, source, success, perturbation_type,
                        perturbation_N, perturbation_position, etc.
        """
        query = {"event": "profiling_run", **filters}
        return list(self.logger.query_with_metadata(query))

    def get_run_features(self, run_id: str, feature_type: Optional[str] = None,
                         group_resolution: Optional[str] = None) -> list:
        """Get all scalar features logged for a run.

        Args:
            run_id: The run to query
            feature_type: Filter by feature type (e.g. "attention")
            group_resolution: Filter by "coarse" or "fine"
        """
        query = {"event": "profiling_features", "run_id": run_id}
        if feature_type:
            query["feature_type"] = feature_type
        results = list(self.logger.query_with_metadata(query))
        if group_resolution:
            results = [r for r in results
                       if r["metadata"].get("group_resolution") == group_resolution]
        return results

    def get_run_tensors(self, run_id: str) -> list:
        """Get tensor paths for a run."""
        query = {"event": "profiling_tensor", "run_id": run_id}
        return list(self.logger.query_with_metadata(query))

    def get_run_transcript(self, run_id: str) -> Optional[Dict]:
        """Get the conversation transcript for a run."""
        results = list(self.logger.query_with_metadata(
            {"event": "profiling_transcript", "run_id": run_id}
        ))
        return results[0]["object"] if results else None

    def get_perturbations_of(self, parent_run_id: str) -> list:
        """Get all perturbations derived from a base run."""
        return list(self.logger.query_with_metadata(
            {"event": "profiling_run", "parent_run_id": parent_run_id}
        ))

    def get_cell_stats(self, parent_run_id: str) -> list:
        """Get cell-level stats for a baseline run's perturbation experiment."""
        return list(self.logger.query_with_metadata(
            {"event": "perturbation_cell_stats", "parent_run_id": parent_run_id}
        ))

    def get_distances(self, run_id: str) -> list:
        """Get distance measurements for a perturbation run."""
        return list(self.logger.query_with_metadata(
            {"event": "profiling_distances", "run_id": run_id}
        ))

    def count_runs_for_cell(self, parent_run_id: str,
                            perturbation_type: str,
                            N: int, position: str) -> int:
        """Count how many perturbation runs exist for a specific grid cell.

        Useful for resumability — skip cells that are already complete.
        """
        runs = self.get_perturbations_of(parent_run_id)
        count = 0
        for r in runs:
            meta = r["metadata"]
            if (meta.get("perturbation_type") == perturbation_type
                    and meta.get("perturbation_N") == N
                    and meta.get("perturbation_position") == position):
                count += 1
        return count