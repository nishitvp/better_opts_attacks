import typing

import torch
import transformers

import algorithms.losses_experimental as losses_experimental
import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger


def compute_trajectory_metrics(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    current_tokens: torch.Tensor,
    masks_data: typing.Dict[str, torch.Tensor],
    logger: experiment_logger.ExperimentLogger,
) -> typing.Dict[str, typing.Optional[float]]:
    """
    Per-step progress metrics for phase-transition analysis.

    Uses bulk_forward_iter (the same path as attention_metricized_v2_true_loss)
    to get both logits and attentions in a single forward pass, then computes:

        payload_attn_uniform  — sum of attention to payload across all layers/heads,
                                uniform weighting, matching what Phase-1 optimizes.
        payload_attn_weighted — same but weighted by CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ
                                (the sensitivity weights from clip_cached_abs_grad_dolly_layer_weights).
                                None until Phase-1 has run its first signal step.
        first_token_logprob   — log P(first target token | current_tokens).

    The indexing and weight format mirrors attention_metricized_v2_true_loss /
    pointwise_sum_of_differences_payload_only exactly.
    """
    target_mask: torch.Tensor = masks_data["target_mask"]
    payload_mask: torch.Tensor = masks_data["payload_mask"]
    input_points = current_tokens.unsqueeze(0)  # (1, seq_len)

    payload_attn_uniform = None
    payload_attn_weighted = None
    first_token_logprob = None

    # One forward pass via bulk_forward_iter — same path as attention_metricized_v2_true_loss
    for logits, attentions in attack_utility.bulk_forward_iter(model, input_points):
        # attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
        # Stack and slice to target prediction positions — mirrors attention_metricized_v2_true_loss
        true_attentions = torch.stack([
            a[:, :, -(len(target_mask) + 1):-1, :]
            for a in attentions
        ])  # (n_layers, 1, n_heads, n_target, seq_len)

        attn_to_payload = true_attentions[:, :, :, :, payload_mask]
        # (n_layers, 1, n_heads, n_target, n_payload)

        # Uniform scalar: sum over all dims
        payload_attn_uniform = attn_to_payload.sum().item()

        # Per-head signature: (n_layers, n_heads) — sum out batch/target/payload dims.
        # Free to compute since attn_to_payload is already sliced.
        # Lets us cross-reference with causal-spark / profiler head rankings later.
        head_payload_attn = attn_to_payload.sum(dim=(1, 3, 4)).cpu().numpy()
        # shape: (n_layers, n_heads) — ~1 KB for 32-layer models, trivial to store

        # Sensitivity-weighted scalar: available once Phase-1 signal_function has run
        if losses_experimental.CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ is not None:
            # CLIPPED_CACHED shape: (n_layers, n_heads) — broadcast to (n_layers, 1, n_heads, 1, 1)
            w = (
                losses_experimental.CLIPPED_CACHED_DOLLY_LAYER_WEIGHT_OBJ
                .to(attn_to_payload.device)
                .unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            )
            payload_attn_weighted = (attn_to_payload * w).sum().item()

        # First target token logprob from the same logit tensor.
        # position -(len(target_mask)+1) predicts target_mask[0] — same convention as target_logprobs
        first_pred_logits = logits[0, -(len(target_mask) + 1), :].float()
        log_probs = torch.nn.functional.log_softmax(first_pred_logits, dim=-1)
        first_token_logprob = log_probs[current_tokens[target_mask[0]].item()].item()

        break  # single input, one batch

    return {
        "payload_attn_uniform": payload_attn_uniform,
        "payload_attn_weighted": payload_attn_weighted,
        "head_payload_attn": head_payload_attn,   # (n_layers, n_heads) numpy array
        "first_token_logprob": first_token_logprob,
    }
