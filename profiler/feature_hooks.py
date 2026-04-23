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
    capture_logit_lens: bool = False     # project hidden states through lm_head (top-k)
    capture_value_aggregates: bool = False  # sum_i(alpha_i * v_i) per head before o_proj
    capture_mlp_output: bool = False        # MLP/MoE contribution to residual stream

    # For attention, capture only the last row (attention pattern for the
    # last token, which is what drives generation). Saves memory.
    attention_last_row_only: bool = True

    # If set, only capture features at specific token positions
    # (e.g., the generated tool-call tokens). None = all positions.
    capture_positions: Optional[List[int]] = None

    # Device for stored tensors (cpu to save GPU memory)
    storage_device: str = "cpu"

    # Number of top tokens to store for logit lens (per position per layer)
    logit_lens_top_k: int = 50

    def should_capture_layer(self, layer_idx: int) -> bool:
        if self.layers is None:
            return True
        return layer_idx in self.layers


@dataclass
class CapturedFeatures:
    """Container for features captured during a forward pass."""

    # Attention weights: {layer_idx: tensor}
    # Shape per tensor: (num_heads, seq_len) if last_row_only
    #                   (num_heads, num_positions, seq_len) if capture_positions set
    attention: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Hidden states (residual stream output) at each profiled layer: {layer_idx: tensor}
    # Shape: (seq_len, hidden_dim) or (num_positions, hidden_dim)
    hidden_states: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Logit lens: top-k projections of hidden states through lm_head.
    # Populated by compute_logit_lens() after the forward pass.
    # {layer_idx: {"top_tokens": (num_positions, k), "top_logits": (num_positions, k)}}
    logit_lens: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)

    # Attention sink mass: probability mass that did NOT go to any real key position
    # (model-specific learned sink scalar). {layer_idx: tensor}
    # Shape: (num_heads,) if last_row_only, (num_heads, num_positions) otherwise
    sink_mass: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Value aggregates: sum_i(alpha_i * v_i) for each head, before the output
    # projection. Captures WHAT information is retrieved, not just WHERE.
    # {layer_idx: tensor}, shape: (num_positions, hidden_dim)
    # Reshape to (num_positions, num_heads, head_dim) in analysis.
    value_aggregates: Dict[int, torch.Tensor] = field(default_factory=dict)

    # MLP (or MoE block) contribution to residual stream: {layer_idx: tensor}
    # Shape: (num_positions, hidden_dim)
    mlp_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        self.attention.clear()
        self.hidden_states.clear()
        self.logit_lens.clear()
        self.sink_mass.clear()
        self.value_aggregates.clear()
        self.mlp_outputs.clear()


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

    def _find_o_proj(self, attn_module: nn.Module) -> Optional[nn.Module]:
        """Find the output projection within an attention module."""
        for name in ["o_proj", "out_proj", "dense"]:
            if hasattr(attn_module, name):
                return getattr(attn_module, name)
        return None

    def _find_mlp_module(self, layer: nn.Module) -> Optional[nn.Module]:
        """Find the MLP/MoE sub-module within a layer."""
        for name in ["mlp", "block_sparse_moe", "moe", "ff", "ffn", "feed_forward"]:
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

            attn_module = self._find_attn_module(layer_module)

            # Attention weights hook
            if self.config.capture_attention and attn_module is not None:
                hook = attn_module.register_forward_hook(
                    self._make_attention_hook(layer_idx)
                )
                self._hooks.append(hook)

            # Value aggregate hook: pre-hook on o_proj captures its input,
            # which is the concatenated per-head value aggregates sum_i(alpha_i v_i)
            if self.config.capture_value_aggregates and attn_module is not None:
                o_proj = self._find_o_proj(attn_module)
                if o_proj is not None:
                    hook = o_proj.register_forward_pre_hook(
                        self._make_value_aggregate_hook(layer_idx)
                    )
                    self._hooks.append(hook)
                else:
                    logger.warning("Layer %d: no o_proj found, skipping value aggregates", layer_idx)

            # MLP/MoE output hook
            if self.config.capture_mlp_output:
                mlp_module = self._find_mlp_module(layer_module)
                if mlp_module is not None:
                    hook = mlp_module.register_forward_hook(
                        self._make_mlp_hook(layer_idx)
                    )
                    self._hooks.append(hook)
                else:
                    logger.warning("Layer %d: no MLP module found, skipping mlp_output", layer_idx)

            # Hidden state hook (full layer output = residual stream after attn+mlp)
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
        config = self.config
        features = self.features

        def hook_fn(module, input, output):
            attn_weights = None
            if isinstance(output, tuple) and len(output) >= 2:
                candidate = output[1]
                if isinstance(candidate, torch.Tensor) and candidate.dim() >= 3:
                    attn_weights = candidate

            if attn_weights is None:
                for attr in ["_attn_weights", "attn_weights", "attention_weights"]:
                    if hasattr(module, attr):
                        attn_weights = getattr(module, attr)
                        break

            if attn_weights is None:
                return

            if attn_weights.dim() == 4:
                attn_weights = attn_weights[0]

            # Compute per-head sink mass: 1 - sum(real_attention_weights)
            # This is the probability mass that went to the learned sink scalar
            if config.attention_last_row_only:
                sink_mass = 1.0 - attn_weights[:, -1, :].sum(dim=-1)  # (num_heads,)
                attn_weights = attn_weights[:, -1, :]
            else:
                sink_mass = 1.0 - attn_weights.sum(dim=-1)  # (num_heads, seq_len)

            if config.capture_positions is not None:
                if attn_weights.dim() == 3:
                    attn_weights = attn_weights[:, config.capture_positions, :]
                    sink_mass = sink_mass[:, config.capture_positions]

            features.attention[layer_idx] = attn_weights.to(config.storage_device).detach()
            features.sink_mass[layer_idx] = sink_mass.to(config.storage_device).detach()

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

    def _make_value_aggregate_hook(self, layer_idx: int):
        """Create a pre-hook on o_proj that captures the value aggregate.

        The input to o_proj is the concatenated per-head outputs:
            sum_i(alpha_i * v_i) for each head, reshaped to (batch, seq_len, hidden_dim).
        This is WHAT gets written to the residual stream per head, before the output
        projection mixes heads together.
        Shape stored: (num_positions, hidden_dim) — reshape to (num_positions, num_heads,
        head_dim) in analysis using num_heads from the attention tensor.
        """
        config = self.config
        features = self.features

        def pre_hook_fn(module, args):
            x = args[0]  # (batch, seq_len, hidden_dim)
            if x.dim() == 3:
                x = x[0]  # (seq_len, hidden_dim)
            if config.capture_positions is not None:
                x = x[config.capture_positions]  # (num_positions, hidden_dim)
            features.value_aggregates[layer_idx] = x.to(config.storage_device).detach()

        return pre_hook_fn

    def _make_mlp_hook(self, layer_idx: int):
        """Create a forward hook that captures the MLP/MoE contribution to the residual stream.

        This is the additive contribution from the feed-forward sublayer, separate from
        the attention sublayer's contribution. JailbreakLens (2024) found MLP contributions
        are the primary driver of jailbreak amplification, not attention routing.
        Shape stored: (num_positions, hidden_dim).
        """
        config = self.config
        features = self.features

        def hook_fn(module, input, output):
            # MoE output may be (hidden_states, router_logits) or just hidden_states
            mlp_out = output[0] if isinstance(output, tuple) else output
            if mlp_out.dim() == 3:
                mlp_out = mlp_out[0]  # (seq_len, hidden_dim)
            if config.capture_positions is not None:
                mlp_out = mlp_out[config.capture_positions]
            features.mlp_outputs[layer_idx] = mlp_out.to(config.storage_device).detach()

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
        top_k: int = 50,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Project intermediate hidden states through the LM head (logit lens).

        For each captured layer and each captured position, applies the final
        layer norm (if provided) and the LM head. Reveals when, layer by layer,
        the model's working prediction commits to injection-task vs user-task tokens.

        Args:
            lm_head:    The model's output projection (model.lm_head).
            layer_norm: The final layer norm (model.model.norm). Applied before
                        lm_head. If None, projects raw hidden states.
            top_k:      Number of top tokens to store per position per layer.

        Returns:
            {layer_idx: {
                "top_tokens": tensor(num_positions, top_k),   # token IDs
                "top_logits": tensor(num_positions, top_k),   # raw logits (not probs)
            }}
        Raw logits are stored rather than probabilities so callers can compute
        gaps/differences between specific token IDs without softmax distortion.
        """
        if not self.features.hidden_states:
            logger.warning("No hidden states captured. Run a forward pass first.")
            return {}

        lm_device = next(lm_head.parameters()).device
        results = {}
        with torch.no_grad():
            for layer_idx, hidden in self.features.hidden_states.items():
                # hidden: (num_positions, hidden_dim) after position filtering
                h = hidden.to(lm_device)
                if layer_norm is not None:
                    h = layer_norm(h)
                logits = lm_head(h)  # (num_positions, vocab_size)
                top_logits, top_ids = logits.topk(top_k, dim=-1)
                results[layer_idx] = {
                    "top_tokens": top_ids.cpu(),
                    "top_logits": top_logits.cpu(),
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