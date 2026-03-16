"""
profiling_logger.py
====================

Thin domain layer on top of ExperimentLogger (the SQL-based one).

Does NOT replace ExperimentLogger. Uses it as the storage backend and adds:
- Structured logging for profiling runs (attack + context + outcome + features)
- Querying by domain-specific fields (suite, success, perturbation type, etc.)
- Tensor storage on disk with paths tracked in metadata
- Perturbation chain tracking via parent_run_id

Everything stored through the underlying ExperimentLogger is still accessible
via the standard logger.query() / logger.query_with_metadata() interface.
The methods here are just convenience wrappers that tag things consistently.

Usage:
    from experiment_logger_sql import ExperimentLogger  # your existing logger
    
    logger = ExperimentLogger("profiling_logs/")
    profiler = ProfilingLogger(logger, tensor_dir="profiling_logs/tensors/")
    
    # Log a run
    run_id = profiler.log_run(
        attack_string="Ignore previous instructions...",
        context={"suite": "workspace", "user_task_id": "user_task_0", ...},
        outcome={"success": True, "reward": 0.85},
        source="rlhammer",
    )
    
    # Log features for that run
    profiler.log_scalar_features(run_id, layer=12, head=7,
        payload_attention_mass=0.83, user_attention_mass=0.12)
    
    # Log large tensors
    profiler.log_tensor(run_id, "attn_layer12", some_tensor)
    
    # Query
    runs_df = profiler.query_successful_runs(suite="workspace")
"""

import json
import uuid
import time
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

# Import your existing logger — adjust the import path as needed
# from experiment_logger_sql import ExperimentLogger


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
        """
        Log a profiling run.
        
        Args:
            attack_string: The injection text
            context: AgentDojo context dict with keys like:
                suite, user_task_id, user_task_prompt, 
                injection_task_id, injection_goal
            outcome: Dict with success, reward, loo_user_delta, etc.
            source: Origin of this attack ("rlhammer", "astra", "manual", "perturbed")
            target_model: Model being attacked
            parent_run_id: If this is a perturbation of another run
            perturbation: Dict with perturbation_type, magnitude, axis
            transcript_dict: Serialized ConversationTranscript (from .to_dict())
            
        Returns:
            run_id (str)
        """
        run_id = str(uuid.uuid4())
        
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
            injection_goal=context.get("injection_goal"),
            # Outcome fields
            success=outcome.get("success") if outcome else None,
            reward=outcome.get("reward") if outcome else None,
            loo_user_delta=outcome.get("loo_user_delta") if outcome else None,
            loo_injection_delta=outcome.get("loo_injection_delta") if outcome else None,
            dominance_shift_margin=outcome.get("dominance_shift_margin") if outcome else None,
            # Perturbation fields
            perturbation_type=perturbation.get("type") if perturbation else "none",
            perturbation_magnitude=perturbation.get("magnitude") if perturbation else None,
            perturbation_axis=perturbation.get("axis") if perturbation else None,
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
        """
        Log scalar features for a specific (layer, head) at a specific step.
        
        Common feature keys:
            payload_attention_mass, user_attention_mass, system_attention_mass,
            sensitivity, residual_norm, logit_lens_target_prob, ...
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
    # 
    # These use the underlying logger's query interface.
    # For complex queries, you can always fall back to:
    #   logger.query_with_metadata({"event": "profiling_run", "suite": "workspace"})
    
    def get_runs(self, **filters) -> list:
        """
        Get profiling runs matching filters.
        
        Filters are passed as metadata key-value pairs.
        Common filters: suite, source, success, target_model, etc.
        
        Returns list of {object, metadata} dicts.
        """
        query = {"event": "profiling_run", **filters}
        return list(self.logger.query_with_metadata(query))
    
    def get_run_features(self, run_id: str, feature_type: Optional[str] = None) -> list:
        """Get all scalar features logged for a run."""
        query = {"event": "profiling_features", "run_id": run_id}
        if feature_type:
            query["feature_type"] = feature_type
        return list(self.logger.query_with_metadata(query))
    
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