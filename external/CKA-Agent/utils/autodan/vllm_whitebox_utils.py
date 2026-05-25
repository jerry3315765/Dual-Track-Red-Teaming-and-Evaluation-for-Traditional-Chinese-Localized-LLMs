#!/usr/bin/env python3
"""
vLLM Whitebox Mode Utilities for AutoDAN
Implements fitness calculation using vLLM's top-k logprobs instead of full vocab logits
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def get_score_autodan_vllm(
    tokenizer,
    conv_template,
    instruction: str,
    target: str,
    model,
    device: str,
    test_controls: List[str],
    crit=None,
    logprob_topk: int = 10000,
    logprob_epsilon: float = -50.0,
) -> torch.Tensor:
    """
    Calculate AutoDAN fitness scores using vLLM's top-k logprobs.

    This is an approximation of the original get_score_autodan that uses full vocab logits.
    When target tokens are not in top-k, we use epsilon as a conservative estimate.

    Args:
        tokenizer: Tokenizer for the model
        conv_template: Conversation template
        instruction: Original harmful query
        target: Target response
        model: vLLM model instance
        device: Device string (for compatibility)
        test_controls: List of adversarial prompts to evaluate
        crit: Loss criterion (CrossEntropyLoss)
        logprob_topk: Number of top logprobs to return from vLLM
        logprob_epsilon: Epsilon value for missing target tokens

    Returns:
        torch.Tensor: Fitness scores (losses) for each test_control
    """
    if crit is None:
        crit = torch.nn.CrossEntropyLoss(reduction="mean")

    losses = []

    for adv_string in test_controls:
        try:
            # Create suffix manager for this adversarial string
            from utils.autodan.string_utils import autodan_SuffixManager

            suffix_manager = autodan_SuffixManager(
                tokenizer=tokenizer,
                conv_template=conv_template,
                instruction=instruction,
                target=target,
                adv_string=adv_string,
            )

            # Get input IDs and target slice
            input_ids = suffix_manager.get_input_ids(adv_string=adv_string)
            target_slice = suffix_manager._target_slice  # Convert to list for vLLM
            input_ids_list = input_ids.tolist()

            # Generate with vLLM to get logprobs
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for fitness calculation
                max_tokens=1,  # We only need next token probabilities
                logprobs=logprob_topk,  # Get top-k logprobs
                prompt_logprobs=logprob_topk,  # Get logprobs for input tokens too
            )

            # Generate with vLLM - ensure input is properly formatted
            try:
                # Convert input_ids_list to string if it's not already
                if isinstance(input_ids_list, list) and len(input_ids_list) > 0:
                    # Decode the token IDs to string for vLLM
                    input_text = tokenizer.decode(
                        input_ids_list, skip_special_tokens=False
                    )
                    outputs = model.generate([input_text], sampling_params)
                else:
                    raise ValueError("Invalid input_ids_list format")

            except Exception as e:
                raise RuntimeError(f"EARLY STOP: vLLM generation failed: {e}")
                # Fallback: try with token IDs directly
                try:
                    outputs = model.generate([input_ids_list], sampling_params)
                except Exception as e2:
                    raise e2
            output = outputs[0]

            # Extract logprobs for target positions
            target_losses = []
            for pos in range(target_slice.start, target_slice.stop):
                if pos < len(output.prompt_logprobs):
                    # Get logprobs for this position
                    logprobs = output.prompt_logprobs[pos]

                    # Convert to tensor format for loss calculation
                    logits_approx = _convert_logprobs_to_logits(
                        logprobs, tokenizer, logprob_epsilon, device
                    )

                    # Get target token ID
                    target_token_id = input_ids[pos].item()

                    # Calculate loss for this position
                    loss = _calculate_position_loss(
                        logits_approx, target_token_id, crit, device
                    )
                    target_losses.append(loss)

            # Average loss across target positions
            if target_losses:
                avg_loss = torch.stack(target_losses).mean()
                losses.append(avg_loss)
            else:
                # Fallback: use high loss if no valid positions
                losses.append(torch.tensor(10.0, device=device))

        except Exception as e:
            logger.warning(f"Error calculating fitness for '{adv_string[:50]}...': {e}")
            # Fallback: use high loss for failed cases
            losses.append(torch.tensor(10.0, device=device))

    return torch.stack(losses)


def _convert_logprobs_to_logits(
    logprobs: Dict[int, float], tokenizer, epsilon: float, device: str
) -> torch.Tensor:
    """
    Convert vLLM logprobs to approximate logits tensor.
    Handles both dict[int->float] and list[Logprob] objects from vLLM.
    """
    vocab_size = len(tokenizer)
    logits = torch.full((vocab_size,), epsilon, device=device, dtype=torch.float32)

    if logprobs is None:
        return logits

    try:
        # vLLM may return a list of Logprob objects
        if isinstance(logprobs, (list, tuple)):
            for lp in logprobs:
                # Try common attribute names
                token_id = getattr(lp, "token_id", None)
                if token_id is None:
                    token_id = getattr(lp, "id", None)
                value = getattr(lp, "logprob", None)
                if (
                    token_id is not None
                    and value is not None
                    and 0 <= int(token_id) < vocab_size
                ):
                    logits[int(token_id)] = float(value)
        elif isinstance(logprobs, dict):
            # dict of token_id -> logprob
            for token_id, value in logprobs.items():
                try:
                    tid = int(token_id)
                except Exception:
                    # token_id might be a token string; try to convert via tokenizer
                    try:
                        # Best effort: get id from token string
                        tid = tokenizer.convert_tokens_to_ids(token_id)
                    except Exception:
                        tid = -1
                if tid is not None and 0 <= tid < vocab_size:
                    # value may be Logprob object or float
                    val = float(getattr(value, "logprob", value))
                    logits[tid] = val
        else:
            # Unknown format; leave as epsilon-filled
            pass
    except Exception as e:
        logger.debug(f"_convert_logprobs_to_logits fallback due to: {e}")
        # Leave logits as epsilon-filled tensor

    return logits


def _calculate_position_loss(
    logits: torch.Tensor, target_token_id: int, crit: torch.nn.Module, device: str
) -> torch.Tensor:
    """
    Calculate loss for a single position using approximate logits.

    Args:
        logits: Approximate logits [vocab_size]
        target_token_id: Target token ID
        crit: Loss criterion
        device: Device string

    Returns:
        torch.Tensor: Loss value
    """
    # Reshape for CrossEntropyLoss: [1, vocab_size] and [1]
    logits_reshaped = logits.unsqueeze(0)  # [1, vocab_size]
    target = torch.tensor([target_token_id], device=device)  # [1]

    return crit(logits_reshaped, target)


def get_score_autodan_hf(
    tokenizer,
    conv_template,
    instruction: str,
    target: str,
    model,
    device: str,
    test_controls: List[str],
    crit=None,
) -> torch.Tensor:
    """
    Calculate AutoDAN fitness scores using HuggingFace model (original AutoDAN method).

    This maintains full compatibility with the original AutoDAN implementation.

    Args:
        tokenizer: Tokenizer for the model
        conv_template: Conversation template
        instruction: Original harmful query
        target: Target response
        model: HuggingFace model instance
        device: Device string
        test_controls: List of adversarial prompts to evaluate
        crit: Loss criterion (CrossEntropyLoss)

    Returns:
        torch.Tensor: Fitness scores (losses) for each test_control
    """
    # Import original AutoDAN function (prefer local utils path, fallback to external AutoDAN if available)
    try:
        from utils.autodan.opt_utils import get_score_autodan
    except Exception:
        try:
            from AutoDAN.utils.opt_utils import get_score_autodan
        except Exception as e:
            raise ImportError(
                f"Failed to import get_score_autodan from both utils.autodan.opt_utils and AutoDAN.utils.opt_utils: {e}"
            )

    return get_score_autodan(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=instruction,
        target=target,
        model=model,
        device=device,
        test_controls=test_controls,
        crit=crit,
    )
