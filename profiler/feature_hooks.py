"""
feature_hooks.py
================

Forward hook management for extracting internal features from GPT-OSS
during profiling passes.

GPT-OSS is MoE (Mixture of Experts), but attention is shared across all
tokens (only the FFN is expert-routed). So attention hooks work the same
as for dense transformers.

Architecture:
    1. FeatureConfig — what to capture (which layers, heads, feature types)
    2. FeatureCapture — registers/removes hooks, stores captured tensors
    3. Attention mass computation — given captured attention and token span
       groups, compute per-head attention mass on payload vs user vs system

Capture strategy:
    We do NOT capture during autoregressive generation (too expensive).
    Instead, after generation completes, we run a single post-hoc forward
    pass on the full sequence (prompt + generated) with hooks registered.
    This gives us the same attention patterns as during generation (for
    greedy/deterministic decoding) in a single efficient pass.

Usage:
    config = FeatureConfig(
        layers=[0, 8, 16, 24, 31],
        capture_attention=True,
        capture_hidden_states=True,
        capture_logit_lens=True,
    )
    capture = FeatureCapture(model, config)

    # Run profiling forward pass
    with capture.active():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Extract features
    attn_maps = capture.attention_maps        # {layer: tensor}
    hidden    = capture.hidden_states          # {layer: tensor}
    attn_mass = capture.compute_attention_mass(span_groups, seq_len)
    logit_lens = capture.compute_logit_lens(model)  # needs lm_head
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for which features to capture and at which layers."""

    # Which layers to instrument (0-indexed). If None, capture ALL layers
    # (not recommended for large models — pick every Nth or specific layers).
    layers: Optional[List[int]] = None

    # Feature types to capture
    capture_attention: bool = True       # per-head attention weights
    capture_hidden_states: bool = True   # residual stream at each profiled layer
    capture_logit_lens: bool = False     # project hidden states through lm_head

    # For attention, capture only the last row (attention pattern for the
    # last token, which is what drives generation). Saves memory.
    attention_last_row_only: bool = True

    # If set, only capture features at specific token positions
    # (e.g., the generated tool-call tokens). None = all positions.
    capture_positions: Optional[List[int]] = None

    # Device for stored tensors (cpu to save GPU memory)
    storage_device: str = "cpu"

    def should_capture_layer(self, layer_idx: int) -> bool:
        if self.layers is None:
            return True
        return layer_idx in self.layers


@dataclass
class CapturedFeatures:
    """Container for features captured during a forward pass."""

    # Attention weights: {layer_idx: tensor}
    # Shape per tensor: (num_heads, seq_len) if last_row_only
    #                   (num_heads, seq_len, seq_len) otherwise
    attention: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Hidden states at each profiled layer: {layer_idx: tensor}
    # Shape: (seq_len, hidden_dim) or (num_positions, hidden_dim)
    hidden_states: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Logit lens projections: {layer_idx: tensor}
    # Shape: (seq_len, vocab_size) or (num_positions, vocab_size)
    logit_lens: Dict[int, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        self.attention.clear()
        self.hidden_states.clear()
        self.logit_lens.clear()


class FeatureCapture:
    """
    Manages forward hooks on a transformer model for feature extraction.

    Designed for GPT-OSS but works with any HuggingFace transformer that
    follows the standard architecture:
        model.model.layers[i].self_attn   — attention module
        model.model.layers[i]             — full layer (for residual stream)

    For MoE models, attention is shared (not routed through experts),
    so attention hooks work identically to dense models.
    """

    def __init__(self, model: nn.Module, config: FeatureConfig):
        self.model = model
        self.config = config
        self.features = CapturedFeatures()
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._layer_modules = self._find_layers()

    def _find_layers(self) -> Dict[int, nn.Module]:
        """
        Locate the transformer layers in the model.

        Tries common attribute paths for HuggingFace models.
        Returns {layer_index: layer_module}.
        """
        layers = None

        # Common paths for HuggingFace models
        for path in [
            "model.layers",        # LLaMA, Mistral, GPT-OSS
            "transformer.h",       # GPT-2, GPT-Neo
            "model.decoder.layers",  # OPT, BART
            "encoder.layer",       # BERT
        ]:
            obj = self.model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                layers = obj
                logger.info("Found transformer layers at: %s (%d layers)", path, len(layers))
                break
            except AttributeError:
                continue

        if layers is None:
            raise ValueError(
                "Could not locate transformer layers. Check model architecture. "
                "Tried: model.layers, transformer.h, model.decoder.layers, encoder.layer"
            )

        return {i: layer for i, layer in enumerate(layers)}

    def _find_attn_module(self, layer: nn.Module) -> Optional[nn.Module]:
        """Find the attention sub-module within a layer."""
        for name in ["self_attn", "attn", "attention"]:
            if hasattr(layer, name):
                return getattr(layer, name)
        return None

    # ── Hook registration ───────────────────────────────────────────

    def register_hooks(self):
        """Register forward hooks on configured layers."""
        self.features.clear()
        self._remove_hooks()

        for layer_idx, layer_module in self._layer_modules.items():
            if not self.config.should_capture_layer(layer_idx):
                continue

            # Attention hook
            if self.config.capture_attention:
                attn_module = self._find_attn_module(layer_module)
                if attn_module is not None:
                    hook = attn_module.register_forward_hook(
                        self._make_attention_hook(layer_idx)
                    )
                    self._hooks.append(hook)

            # Hidden state hook (on the full layer output = residual stream)
            if self.config.capture_hidden_states:
                hook = layer_module.register_forward_hook(
                    self._make_hidden_state_hook(layer_idx)
                )
                self._hooks.append(hook)

        logger.info("Registered %d hooks across %d layers",
                     len(self._hooks),
                     sum(1 for i in self._layer_modules if self.config.should_capture_layer(i)))

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @contextmanager
    def active(self):
        """
        Context manager: register hooks, yield, then clean up.

        Usage:
            with capture.active():
                outputs = model(input_ids)
            # hooks automatically removed, features available in capture.features
        """
        self.register_hooks()
        try:
            yield self.features
        finally:
            self._remove_hooks()

    # ── Hook factories ──────────────────────────────────────────────

    def _make_attention_hook(self, layer_idx: int):
        """Create a forward hook that captures attention weights."""
        config = self.config
        features = self.features

        def hook_fn(module, input, output):
            # HuggingFace attention modules return:
            #   (attn_output, attn_weights, past_key_value)
            # when output_attentions=True. But with hooks, we may need
            # to extract from the module directly.
            attn_weights = None

            # Try to get from output tuple
            if isinstance(output, tuple) and len(output) >= 2:
                candidate = output[1]
                if isinstance(candidate, torch.Tensor) and candidate.dim() >= 3:
                    attn_weights = candidate

            if attn_weights is None:
                # Some models store attention weights as a module attribute
                # during forward (e.g., module._attn_weights)
                for attr in ["_attn_weights", "attn_weights", "attention_weights"]:
                    if hasattr(module, attr):
                        attn_weights = getattr(module, attr)
                        break

            if attn_weights is None:
                logger.debug("Layer %d: attention weights not available in hook output", layer_idx)
                return

            # attn_weights shape: (batch, num_heads, seq_len, seq_len)
            # Remove batch dim (we process one sequence at a time)
            if attn_weights.dim() == 4:
                attn_weights = attn_weights[0]  # (num_heads, seq_len, seq_len)

            if config.attention_last_row_only:
                # Only keep the last row: what the last token attends to
                # Shape: (num_heads, seq_len)
                attn_weights = attn_weights[:, -1, :]

            if config.capture_positions is not None:
                # Only keep specific positions
                if attn_weights.dim() == 3:
                    attn_weights = attn_weights[:, config.capture_positions, :]
                # For last_row_only, all positions are in the seq_len dim (kept)

            features.attention[layer_idx] = attn_weights.to(config.storage_device).detach()

        return hook_fn

    def _make_hidden_state_hook(self, layer_idx: int):
        """Create a forward hook that captures hidden states (residual stream)."""
        config = self.config
        features = self.features

        def hook_fn(module, input, output):
            # Layer output is typically (hidden_states, ...) or just hidden_states
            hidden = output[0] if isinstance(output, tuple) else output

            # hidden shape: (batch, seq_len, hidden_dim)
            if hidden.dim() == 3:
                hidden = hidden[0]  # (seq_len, hidden_dim)

            if config.capture_positions is not None:
                hidden = hidden[config.capture_positions]

            features.hidden_states[layer_idx] = hidden.to(config.storage_device).detach()

        return hook_fn

    # ── Feature computation ─────────────────────────────────────────

    def compute_attention_mass(
        self,
        span_groups: Dict[str, List[Tuple[int, int]]],
        seq_len: int,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Compute attention mass on each span group at each captured layer.

        For each (layer, head), sums the attention weights falling within
        each span group's token ranges.

        Args:
            span_groups: From ConversationTranscript.get_span_groups().
                         Keys like "system", "user", "payload", etc.
                         Values are lists of (start, end) token spans.
            seq_len:     Total sequence length (for mask construction).

        Returns:
            {group_name: {layer_idx: tensor(num_heads)}}
            Each tensor contains the total attention mass for that group
            per attention head.
        """
        if not self.features.attention:
            logger.warning("No attention maps captured. Run a forward pass first.")
            return {}

        # Build boolean masks for each span group
        masks: Dict[str, torch.Tensor] = {}
        for name, spans in span_groups.items():
            mask = torch.zeros(seq_len, dtype=torch.bool)
            for start, end in spans:
                mask[start:end] = True
            masks[name] = mask

        result: Dict[str, Dict[int, torch.Tensor]] = {
            name: {} for name in span_groups
        }

        for layer_idx, attn in self.features.attention.items():
            # attn shape: (num_heads, seq_len) if last_row_only
            #             (num_heads, seq_len, seq_len) otherwise
            for name, mask in masks.items():
                # Ensure mask is on same device as attn
                m = mask.to(attn.device)

                if attn.dim() == 2:
                    # Last-row-only: (num_heads, seq_len)
                    # Mask the key positions
                    mass = (attn * m.unsqueeze(0)).sum(dim=-1)  # (num_heads,)
                elif attn.dim() == 3:
                    # Full attention: (num_heads, q_len, k_len)
                    # Sum over key positions for each query position, then average over queries
                    mass = (attn * m.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean(dim=-1)
                else:
                    logger.warning("Unexpected attention shape: %s", attn.shape)
                    continue

                result[name][layer_idx] = mass.cpu()

        return result

    def compute_logit_lens(
        self,
        lm_head: nn.Module,
        layer_norm: Optional[nn.Module] = None,
        top_k: int = 10,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Project intermediate hidden states through the LM head (logit lens).

        For each captured layer, applies the final layer norm (if provided)
        and the LM head to see what tokens the model "believes" at that layer.

        Args:
            lm_head:    The model's output projection (model.lm_head)
            layer_norm: The final layer norm (model.model.norm), applied before
                        the lm_head. If None, projects raw hidden states.
            top_k:      Number of top tokens to return per layer.

        Returns:
            {layer_idx: {
                "top_tokens": tensor(top_k),       # token IDs
                "top_probs": tensor(top_k),         # probabilities
                "target_probs": dict                # filled in by caller if needed
            }}
        """
        if not self.features.hidden_states:
            logger.warning("No hidden states captured. Run a forward pass first.")
            return {}

        results = {}
        with torch.no_grad():
            for layer_idx, hidden in self.features.hidden_states.items():
                # hidden: (seq_len, hidden_dim) or (num_positions, hidden_dim)
                # Take the last position (what generates the next token)
                h = hidden[-1:]  # (1, hidden_dim)
                h = h.to(next(lm_head.parameters()).device)

                if layer_norm is not None:
                    h = layer_norm(h)

                logits = lm_head(h)  # (1, vocab_size)
                probs = torch.softmax(logits[0], dim=-1)

                top_probs, top_ids = probs.topk(top_k)

                results[layer_idx] = {
                    "top_tokens": top_ids.cpu(),
                    "top_probs": top_probs.cpu(),
                    "full_probs": probs.cpu(),  # for target_prob lookup
                }

        return results

    # ── Scalar feature extraction ───────────────────────────────────

    def extract_scalar_features(
        self,
        span_groups: Dict[str, List[Tuple[int, int]]],
        seq_len: int,
    ) -> List[Dict[str, Any]]:
        """
        Extract scalar features suitable for ProfilingLogger.log_scalar_features().

        Returns a list of dicts, one per (layer, head) combination, with keys:
            layer, head, payload_attention_mass, user_attention_mass,
            system_attention_mass, tool_clean_attention_mass,
            residual_norm (if hidden states captured)

        This is the main interface for the inference runner to get loggable features.
        """
        attn_mass = self.compute_attention_mass(span_groups, seq_len)
        features_list = []

        # Get the set of captured layers
        captured_layers = sorted(
            set(self.features.attention.keys()) | set(self.features.hidden_states.keys())
        )

        for layer_idx in captured_layers:
            # Determine number of heads from attention data
            num_heads = 0
            if layer_idx in self.features.attention:
                num_heads = self.features.attention[layer_idx].shape[0]

            if num_heads == 0:
                # No attention data for this layer, just log residual norm
                entry = {
                    "layer": layer_idx,
                    "head": None,
                    "feature_type": "residual",
                }
                if layer_idx in self.features.hidden_states:
                    h = self.features.hidden_states[layer_idx]
                    entry["residual_norm"] = h[-1].norm().item()
                features_list.append(entry)
                continue

            for head_idx in range(num_heads):
                entry = {
                    "layer": layer_idx,
                    "head": head_idx,
                    "feature_type": "attention",
                }

                # Attention mass per span group
                for group_name in ["system", "user", "payload", "tool_clean", "tool_injected"]:
                    if group_name in attn_mass and layer_idx in attn_mass[group_name]:
                        mass_tensor = attn_mass[group_name][layer_idx]
                        entry[f"{group_name}_attention_mass"] = mass_tensor[head_idx].item()

                # Residual norm (same for all heads at a layer)
                if head_idx == 0 and layer_idx in self.features.hidden_states:
                    h = self.features.hidden_states[layer_idx]
                    entry["residual_norm"] = h[-1].norm().item()

                features_list.append(entry)

        return features_list