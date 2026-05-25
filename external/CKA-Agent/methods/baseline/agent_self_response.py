"""
Agent Self-Response Method Implementation
A baseline method where the attack agent directly answers harmful queries.

Purpose: Evaluate the attack agent's inherent capability without complex jailbreak strategies.
This serves as a baseline to compare against sophisticated methods like CKA-Agent.
"""

from typing import Dict, Any, List, Optional
import time
import json
import logging
import os
import threading
from collections import defaultdict
import re

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.abstract_method import AbstractJailbreakMethod
from model.model_loader import WhiteBoxModel

# vLLM imports
try:
    from vllm import LLM as VLLMEngine, SamplingParams as VLLMSamplingParams

    _VLLM_OK = True
except Exception:
    _VLLM_OK = False


class AgentSelfResponseMethod(AbstractJailbreakMethod):
    """
    Agent Self-Response: Direct answering baseline.

    The attack agent (e.g., Qwen3-32B-abliterated) directly answers harmful queries
    without querying any target LLM. This provides a baseline to measure:
    1. The attack agent's inherent capability
    2. The effectiveness of complex jailbreak strategies (e.g., CKA-Agent)
    """

    def __init__(
        self,
        name: str = "agent_self_response",
        config: Dict[str, Any] = None,
        model=None,
    ):
        """Initialize Agent Self-Response method."""
        default_config = {
            "batch_size": 1,
            "verbose": True,
            "save_intermediate": True,
            "length_control": True,
            "max_response_words": 200,
        }

        if config:
            default_config.update(config)

        super().__init__(name, default_config, model)

        # Load attack agent model (same as CKA-Agent controller)
        attack_cfg = self.config.get("attack_model", {})
        if not attack_cfg.get("name"):
            raise ValueError("attack_model.name must be specified in config")

        # Initialize attack agent with WhiteBoxModel
        self.attack_agent = self._initialize_attack_agent(attack_cfg)

        # Thinking mode configuration
        self.enable_thinking = bool(attack_cfg.get("enable_thinking", False))
        self.remove_thinking = bool(attack_cfg.get("remove_thinking", False))

        # Thread-safe intermediate saver
        self._sample_locks = defaultdict(threading.Lock)
        self._thread_local = threading.local()

        # Output directory
        self.output_dir = getattr(self, "output_dir", os.getcwd())

        self.logger.info(
            f"Initialized Agent Self-Response with model: {self.attack_agent.model_name}"
        )
        self.logger.info(
            f"Thinking mode: enable={self.enable_thinking}, remove={self.remove_thinking}"
        )

    def _initialize_attack_agent(self, config: Dict[str, Any]) -> WhiteBoxModel:
        """
        Initialize attack agent model using WhiteBoxModel.

        Args:
            config: Attack model configuration

        Returns:
            Initialized WhiteBoxModel instance
        """
        # Prepare WhiteBoxModel configuration
        model_config = {
            "use_vllm": config.get("use_vllm", False),
            "vllm_kwargs": config.get("vllm_kwargs", {}),
            "device_map": config.get("device_map", "auto"),
            "max_length": config.get("max_new_tokens", 2048),
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "do_sample": config.get("do_sample", True),
            "hf_token": config.get("hf_token"),
            "controller_compat": True,  # Use controller-compatible settings
            "input_max_length": int(config.get("input_max_length", 131072)),
        }

        # Initialize WhiteBoxModel
        attack_agent = WhiteBoxModel(config.get("name", ""), model_config)
        attack_agent.load(config.get("hf_token"))

        return attack_agent

    def _apply_thinking_mode(self, response: str) -> str:
        """
        Apply thinking mode processing to response.

        If remove_thinking=True, removes <think>...</think> tags from response.

        Args:
            response: Raw response from model

        Returns:
            Processed response with thinking removed if configured
        """
        if self.remove_thinking and "<think>" in response:
            # Remove DeepSeek-style thinking tags
            processed = re.sub(
                r"<think>.*?</think>", "", response, flags=re.DOTALL
            ).strip()
            self.logger.debug(
                f"Removed thinking tags: {len(response)} -> {len(processed)} chars"
            )
            return processed

        return response

    def _build_prompt(self, query: str) -> str:
        """
        Build prompt for attack agent.

        Optionally adds length control instruction to prevent overly long responses.

        Args:
            query: Original harmful query

        Returns:
            Formatted prompt string
        """
        prompt = query

        # Add length control instruction if enabled
        if self.config.get("length_control", True):
            max_words = self.config.get("max_response_words", 200)
            length_instruction = (
                f"\n\nProvide a answer to the prompt, "
                f"No more than {max_words} words."
            )
            prompt = query + length_instruction

        return prompt

    def _chat_with_agent(self, messages: List[Dict[str, str]]) -> str:
        """
        Send messages to attack agent and get response.

        Handles thinking mode and uses tokenizer's chat template.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Agent's response text
        """
        # Apply chat template with thinking mode
        prompt_text = self.attack_agent.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,  # Control thinking mode
        )

        # Generate response using vLLM or HF
        if self.attack_agent.use_vllm and self.attack_agent.vllm_model is not None:
            # vLLM generation
            sampling_params = VLLMSamplingParams(
                max_tokens=self.attack_agent.max_length,
                temperature=self.attack_agent.temperature,
                top_p=self.attack_agent.top_p,
            )

            outputs = self.attack_agent.vllm_model.generate(
                [prompt_text], sampling_params
            )
            response = outputs[0].outputs[0].text.strip()
        else:
            # HF model generation
            response = self.attack_agent._generate_hf(prompt_text)

        # Apply thinking mode processing
        response = self._apply_thinking_mode(response)

        return response

    def _save_intermediate(self, data: Dict[str, Any]) -> None:
        """
        Save intermediate results for debugging/analysis.

        Args:
            data: Data to save
        """
        try:
            sample_index = getattr(self._thread_local, "sample_index", 0)
            dataset_key = getattr(self._thread_local, "dataset_key", "unknown")

            with self._sample_locks[sample_index]:
                os.makedirs(self.output_dir, exist_ok=True)
                inter_path = os.path.join(
                    self.output_dir, f"inter_result_sample_{sample_index}.json"
                )

                payload = {
                    "method": self.name,
                    "sample_index": sample_index,
                    "dataset_key": dataset_key,
                    "timestamp": time.time(),
                    "data": data,
                }

                with open(inter_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    f.flush()

                self.logger.debug(
                    f"Saved intermediate results for sample {sample_index}"
                )

        except Exception as e:
            self.logger.debug(f"Failed to save intermediate results: {e}")

    def generate_jailbreak(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate response by directly querying attack agent.

        Args:
            query: Original harmful query
            **kwargs: Additional parameters

        Returns:
            Result dict containing response and metadata
        """
        start_time = time.time()

        result = {
            "original_query": query,
            "jailbreak_prompt": None,
            "response": None,
            "success": False,
            "metadata": {
                "method": self.name,
                "timestamp": start_time,
                "processing_time": 0,
                "thinking_mode_enabled": self.enable_thinking,
                "thinking_removed": self.remove_thinking,
                "length_control": self.config.get("length_control", True),
                "category": kwargs.get("category", "unknown"),
                "source": kwargs.get("source", "unknown"),
            },
            "error": None,
        }

        try:
            # Build prompt with optional length control
            prompt = self._build_prompt(query)

            # Prepare messages for chat template
            messages = [{"role": "user", "content": prompt}]

            # Get response from attack agent
            response = self._chat_with_agent(messages)

            # Store results
            result["jailbreak_prompt"] = response
            result["response"] = response
            result["success"] = True

            # Save intermediate results
            if self.config.get("save_intermediate", False):
                self._save_intermediate(
                    {
                        "query": query,
                        "prompt": prompt,
                        "response": response,
                        "response_length_chars": len(response),
                        "response_length_words": len(response.split()),
                    }
                )

            # Update stats
            self.update_stats(success=True, error=False)

        except Exception as e:
            error_msg = f"Error in Agent Self-Response: {str(e)}"
            self.logger.error(error_msg)
            import traceback

            self.logger.error(traceback.format_exc())

            result["error"] = error_msg
            result["success"] = False
            self.update_stats(success=False, error=True)

        # Calculate processing time
        result["metadata"]["processing_time"] = time.time() - start_time
        self.logger.info(
            f"Processing time: {result['metadata']['processing_time']:.2f}s"
        )

        return result

    def prepare_prompt(self, query: str, **kwargs) -> str:
        """This method does not use traditional prompt preparation."""
        return query

    def generate_jailbreak_batch(
        self, queries: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch processing for multiple queries.

        Args:
            queries: List of harmful queries
            **kwargs: Additional parameters including base_index and dataset_key

        Returns:
            List of result dicts
        """
        batch_size = self.config.get("batch_size", 1)
        base_index = kwargs.get("base_index", 0)
        dataset_key = kwargs.get("dataset_key", "unknown")

        results = []

        # Process sequentially (batch_size=1 for this simple method)
        for i, query in enumerate(queries):
            # Set thread-local storage for intermediate saving
            self._thread_local.sample_index = base_index + i
            self._thread_local.dataset_key = dataset_key

            result = self.generate_jailbreak(query, **kwargs)
            results.append(result)

        return results

    def teardown(self) -> None:
        """Release resources to free GPU/CPU memory."""
        try:
            self.logger.info("[Agent Self-Response] Releasing resources...")

            # Release attack agent model
            if hasattr(self, "attack_agent") and self.attack_agent is not None:
                # Release vLLM engine or HF model
                for attr in ["vllm_model", "model", "tokenizer"]:
                    if hasattr(self.attack_agent, attr):
                        setattr(self.attack_agent, attr, None)
                self.attack_agent = None

            # Force garbage collection
            import gc
            import torch

            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            self.logger.info("[Agent Self-Response] Resources released successfully")

        except Exception as e:
            self.logger.warning(f"[Agent Self-Response] Teardown warning: {e}")
