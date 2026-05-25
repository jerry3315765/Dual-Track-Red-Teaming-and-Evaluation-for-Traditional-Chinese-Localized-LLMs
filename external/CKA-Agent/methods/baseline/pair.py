from typing import Dict, Any, List, Tuple
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import json
import threading
from collections import defaultdict
import re
import torch
from transformers import GenerationConfig

from methods.abstract_method import AbstractJailbreakMethod
from model.model_loader import WhiteBoxModel, BlackBoxModel

# Import vLLM for accelerated inference
try:
    from vllm import LLM as VLLMEngine, SamplingParams as VLLMSamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# PAIR utils (to be added under utils/pair)
from utils.pair.common import (
    extract_json,
    get_init_msg,
    process_target_response,
    truncate_histories,
)

# Removed: from utils.pair.prompts import get_attacker_system_prompts, get_judge_system_prompt
# Now dynamically imported in _init_conversations (unified text format for all models)
from utils.pair.conversation import make_conv_template, render_full_prompt
from utils.pair.judges import load_pair_judge


class PAIRAttackLLM:
    """PAIR's internal AttackLLM for thinking control (copied from PAPAttackLLM)."""

    def __init__(self, model_name: str, config: Dict, whitebox_model=None):
        self.config = config or {}
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.tokenizer = None
        self.use_vllm = bool(self.config.get("use_vllm", False))
        self.vllm_kwargs = self.config.get("vllm_kwargs", {}) or {}
        self.hf_token = self.config.get("hf_token")
        self.device = self.config.get("device", "cuda")
        self.max_new_tokens = int(
            self.config.get("max_new_tokens", 500)
        )  # Use max_new_tokens from config
        self.temperature = float(self.config.get("temperature", 0.7))
        self.top_p = float(self.config.get("top_p", 0.9))
        self.do_sample = bool(self.config.get("do_sample", True))
        self.max_model_len = int(self.config.get("max_model_len", 4096))
        self.enable_thinking = bool(self.config.get("enable_thinking", False))
        self.remove_thinking = bool(self.config.get("remove_thinking", False))
        self._vllm_engine = None

        # Use pre-loaded WhiteBoxModel (same as PAP)
        if whitebox_model is not None:
            self.logger.info("[PAIRAttack] Using pre-loaded WhiteBoxModel")
            self._use_whitebox_model(whitebox_model)
        else:
            raise ValueError("PAIRAttackLLM requires a pre-loaded WhiteBoxModel")

    def _use_whitebox_model(self, whitebox_model):
        """Use a pre-loaded WhiteBoxModel (same as PAP)."""
        self.tokenizer = whitebox_model.tokenizer
        self.use_vllm = whitebox_model.use_vllm

        if self.use_vllm:
            self._vllm_engine = whitebox_model.vllm_model
            self.logger.info("[PAIRAttack] Using vLLM engine from WhiteBoxModel")
        else:
            self.model = whitebox_model.model
            self.logger.info("[PAIRAttack] Using HF model from WhiteBoxModel")

        # Copy relevant attributes
        self.model_name = whitebox_model.model_name
        self.device = whitebox_model.device
        self.logger.info(
            f"[PAIRAttack] Successfully reused WhiteBoxModel: {self.model_name}"
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat method with thinking control (same as PAP)."""
        content = self._chat(messages)
        if self.remove_thinking:
            # Remove <think>...</think> tags (same as PAP)
            content = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()
        return content

    def _chat(self, messages: List[Dict[str, str]]) -> str:
        """Internal chat method (same as PAP)."""
        prompt = self._messages_to_prompt(messages)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,  # Use config setting
        )

        # Use vLLM if available
        if self._vllm_engine is not None:
            # Use apply_chat_template result for thinking support
            params = VLLMSamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # Use apply_chat_template result which includes thinking support
            outs = self._vllm_engine.generate([prompt_text], params)
            return outs[0].outputs[0].text.strip()
        else:
            # Local HF model processing, use apply_chat_template (same as PAP)
            inputs = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(
                    self.max_model_len, 4096
                ),  # Use the configured maximum length
            )

            # Fix: Automatically detect and use model's device (same as PAP)
            with torch.no_grad():
                # Get model's device from first parameter
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                gen_cfg = GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                outputs = self.model.generate(**inputs, generation_config=gen_cfg)
                input_len = inputs["input_ids"].shape[1]
                text = self.tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                )

                return text.strip()

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string (same as PAP)."""
        buf = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            buf.append(f"{role.upper()}:\n{content}\n")
        buf.append("ASSISTANT:\n")
        return "\n".join(buf)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts."""
        if self.use_vllm and self._vllm_engine is not None:
            return self._generate_vllm_batch(prompts)
        else:
            return [self._generate_hf(prompt) for prompt in prompts]

    def _generate_vllm_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses using vLLM with batch processing."""
        try:
            params = VLLMSamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            outputs = self._vllm_engine.generate(prompts, params)
            responses = []
            for output in outputs:
                response = output.outputs[0].text.strip()
                responses.append(response)

            return responses
        except Exception as e:
            self.logger.error(f"vLLM batch generation failed: {e}")
            return [f"Error: {str(e)}"] * len(prompts)

    def _generate_hf(self, prompt: str) -> str:
        """Generate response using Hugging Face Transformers."""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(self.max_model_len, 4096),
            )

            with torch.no_grad():
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}

                gen_cfg = GenerationConfig(
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                outputs = self.model.generate(**inputs, generation_config=gen_cfg)
                input_len = inputs["input_ids"].shape[1]
                text = self.tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                )

                return text.strip()
        except Exception as e:
            self.logger.error(f"HF generation failed: {e}")
            return f"Error: {str(e)}"


class PairMethod(AbstractJailbreakMethod):
    """
    Integrated PAIR method within the unified framework.
    - Target model: forced blackbox channel
    - Attack model: whitebox (local HF/vLLM) or blackbox (API) per config
    - Online judge: internal to PAIR (separate from evaluation.judge_model)
    """

    def __init__(self, name: str = "pair", config: Dict[str, Any] = None, model=None):
        super().__init__(name, config, model)
        self.logger.setLevel(logging.getLevelName(self.config.get("log_level", "INFO")))
        # Output directory injected by main; fallback to current directory
        self.output_dir = getattr(self, "output_dir", os.getcwd())

        # Thread-safety for per-sample intermediate saves
        self._sample_locks = defaultdict(threading.Lock)
        self._thread_local = threading.local()

        # Load PAIR-specific config pieces
        self.runtime_cfg = self.config.get("runtime", {})
        self.attack_cfg = self.config.get("attack_model", {})
        self.judge_cfg = self.config.get("judge_model", {})
        self.exp_cfg = self.config.get("experiment", {})

        # Runtime params
        self.n_iterations = int(self.runtime_cfg.get("n_iterations", 3))
        self.n_streams = int(self.runtime_cfg.get("n_streams", 3))
        self.keep_last_n = int(self.runtime_cfg.get("keep_last_n", 4))
        self.early_stop_score = int(self.runtime_cfg.get("early_stop_score", 10))
        self.enable_wandb = bool(self.exp_cfg.get("enable_wandb", False))

        # Build attack LM (whitebox or blackbox)
        attack_type = (self.attack_cfg.get("type") or "whitebox").lower()
        self.logger.info(f"PAIR attack_model.type={attack_type}")
        if attack_type == "whitebox":
            wb = self.attack_cfg.get("whitebox", {})
            if not wb.get("name") or str(wb.get("name")).strip() == "":
                raise ValueError(
                    f"PAIR attack whitebox name is empty; attack_model.whitebox={wb}"
                )
            self.attack_is_blackbox = False
            self.attack_initialize_output = bool(wb.get("initialize_output", True))
            self.attack_max_new_tokens = int(wb.get("max_new_tokens", 500))
            self.attack_max_attempts = int(wb.get("max_n_attack_attempts", 5))
            self.attack_temperature = float(wb.get("temperature", 1.0))
            self.attack_top_p = float(wb.get("top_p", 0.9))
            # Determine prompt and parsing mode based on model name
            model_name = wb.get("name", "").lower()

            # All models now use unified text format - no mode detection needed

            # Create WhiteBoxModel (same as PAP)
            whitebox_model = WhiteBoxModel(
                wb.get("name", ""),
                {
                    "use_vllm": wb.get("use_vllm", False),
                    "vllm_kwargs": wb.get("vllm_kwargs", {}),
                    "device_map": wb.get("device_map", None),
                    "max_length": self.attack_max_new_tokens,
                    "temperature": self.attack_temperature,
                    "top_p": self.attack_top_p,
                    "do_sample": True,
                },
            )

            try:
                whitebox_model.load(wb.get("hf_token"))
                self.logger.info(
                    f"WhiteBoxModel loaded successfully: {whitebox_model.model_name}"
                )

                # Create PAIRAttackLLM with the loaded WhiteBoxModel (same as PAP)
                self.attack_lm = PAIRAttackLLM(
                    model_name=whitebox_model.model_name,
                    config=wb,
                    whitebox_model=whitebox_model,  # Pass the loaded WhiteBoxModel
                )

                self.logger.info(
                    f"PAIRAttackLLM initialized with WhiteBoxModel: {self.attack_lm.model_name}"
                )

            except Exception as load_error:
                self.logger.error(f"Failed to load WhiteBoxModel: {load_error}")
                raise

            self.attack_template_name = (
                self.attack_lm.model_name
                if isinstance(self.attack_lm.model_name, str)
                else "gpt-3.5-turbo"
            )
        elif attack_type == "blackbox":
            bb = self.attack_cfg.get("blackbox", {})
            self.logger.info(f"PAIR attack_model.blackbox keys={list(bb.keys())}")
            self.logger.info(f"PAIR attack_model.blackbox.name={bb.get('name')}")
            self.attack_is_blackbox = True
            self.attack_initialize_output = bool(bb.get("initialize_output", True))
            self.attack_max_new_tokens = int(bb.get("max_tokens", 500))
            self.attack_max_attempts = int(bb.get("max_n_attack_attempts", 5))
            self.attack_temperature = float(bb.get("temperature", 1.0))
            self.attack_top_p = float(bb.get("top_p", 0.9))
            # All models now use unified text format - no mode detection needed
            self.attack_lm = BlackBoxModel(bb.get("name", ""), bb)
            self.attack_lm.load()
            self.attack_template_name = "gpt-3.5-turbo"
        else:
            raise ValueError(f"Unsupported attack_model.type: {attack_type}")

        # Target LM is provided by the framework as model
        # Support both whitebox and blackbox target models
        self.target_is_blackbox = isinstance(self.model, BlackBoxModel)
        self.logger.info(
            f"PAIR target model type: {'blackbox' if self.target_is_blackbox else 'whitebox'}"
        )

        # Online judge model for PAIR internal optimization
        self.judge = load_pair_judge(self.judge_cfg)

    def _save_intermediate(
        self,
        iteration: int,
        per_stream_payload: List[Dict[str, Any]],
        best_overall: Dict[str, Any],
    ) -> None:
        """
        Align with AutoDAN: save intermediate results per sample in a single file (JSONL).
        Filename: inter_result_sample_{sample_index}.json, located in output_dir.
        Each line: iteration, streams[{candidate_json, rendered_prompt, target_prompt, response, score}], best_overall.
        Thread-safe: use sample-level locks (inherited from AbstractJailbreakMethod's sample lock dictionary).
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            # Get sample context from thread-local storage
            sample_index = getattr(self._thread_local, "sample_index", 0)
            dataset_key = getattr(self._thread_local, "dataset_key", "unknown")

            # If sample_index is 0, try to infer the correct sample_index from the length of pending_judge.jsonl
            if sample_index == 0:
                # Check if pending_judge.jsonl file exists
                pending_judge_path = os.path.join(
                    self.output_dir, "pending_judge.jsonl"
                )
                if os.path.exists(pending_judge_path):
                    try:
                        # Read the number of lines in pending_judge.jsonl
                        with open(pending_judge_path, "r", encoding="utf-8") as f:
                            pending_count = sum(1 for _ in f)
                        # Use the number of lines in pending_judge.jsonl as sample_index
                        sample_index = pending_count
                        # Update thread-local storage
                        try:
                            self._thread_local.sample_index = sample_index
                        except Exception:
                            pass
                    except Exception:
                        # If reading fails, keep sample_index = 0
                        pass

            # Construct sample-level file path
            inter_path = os.path.join(
                self.output_dir, f"inter_result_sample_{sample_index}.json"
            )
            record = {
                "method": self.name,
                "dataset_key": dataset_key,
                "sample_index": int(sample_index),
                "iteration": int(iteration),
                "streams": per_stream_payload,
                "best_overall": best_overall,
            }
            # Sample-level lock to avoid concurrent write conflicts
            try:
                lock = self._sample_locks[sample_index]
            except Exception:
                lock = None
            if lock is not None:
                with lock:
                    with open(inter_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                with open(inter_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # Non-fatal
            try:
                self.logger.debug(
                    f"PAIR intermediate save failed at iter {iteration}: {e}"
                )
            except Exception:
                pass

    def _init_conversations(
        self, goal: str, target_str: str, template_name: str
    ) -> Tuple[List[Any], List[str]]:
        convs_list = [make_conv_template(template_name) for _ in range(self.n_streams)]

        # Import the unified prompt module
        from utils.pair.prompts import get_attacker_system_prompts

        system_prompts = get_attacker_system_prompts(goal, target_str)

        for i, conv in enumerate(convs_list):
            prompt_index = i % len(system_prompts)
            selected_prompt = system_prompts[prompt_index]
            conv.set_system_message(selected_prompt)
        init_msg = get_init_msg(goal, target_str)
        processed_response_list = [init_msg for _ in range(self.n_streams)]
        return convs_list, processed_response_list

    def _attack_batch(
        self,
        convs_list: List[Any],
        processed_response_list: List[str],
        goal: str,
        target_str: str,
    ) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        """Use attack LM (whitebox/blackbox) to batch-generate attacks for all streams."""

        # All models now use text format - no initialization needed
        for conv, prompt in zip(convs_list, processed_response_list):
            conv.append_message(conv.roles[0], prompt)

        # Render full prompts including system + history using fastchat
        full_prompts: List[str] = [render_full_prompt(conv) for conv in convs_list]
        messages_batches: List[List[Dict[str, Any]]] = [
            conv.to_openai_api_messages() for conv in convs_list
        ]

        if self.attack_is_blackbox:
            outputs = []
            indices_to_regenerate = list(range(len(full_prompts)))
            for attempt in range(self.attack_max_attempts):
                sub_msgs = [messages_batches[i] for i in indices_to_regenerate]
                batch_outputs = self.attack_lm.generate_batch_messages(sub_msgs)
                new_indices = []

                # All models now use text format - no processing needed
                stitched = batch_outputs

                for j, full_output in enumerate(stitched):
                    attack_dict, rendered_prompt = extract_json(full_output)

                    if attack_dict is None or not attack_dict.get("prompt", "").strip():
                        self.logger.warning(
                            f"Parsing failed for stream {indices_to_regenerate[j]}, will retry"
                        )
                        new_indices.append(indices_to_regenerate[j])
                    else:
                        # All models now use text format - use extracted prompt
                        outputs.append(
                            (
                                indices_to_regenerate[j],
                                attack_dict,
                                attack_dict["prompt"],
                                full_output,
                            )
                        )
                indices_to_regenerate = new_indices
                if not indices_to_regenerate:
                    break
            # Handle failed streams with fallback
            if indices_to_regenerate:
                self.logger.warning(
                    f"{len(indices_to_regenerate)} streams failed after {self.attack_max_attempts} attempts, using fallback"
                )
                for idx in indices_to_regenerate:
                    fallback_dict = {"improvement": "", "prompt": ""}
                    outputs.append((idx, fallback_dict, "", ""))

            outputs_sorted = sorted(outputs, key=lambda x: x[0])
            attack_dicts = [x[1] for x in outputs_sorted]
            jailbreak_prompts = [x[2] for x in outputs_sorted]
            rendered_prompts = [x[3] for x in outputs_sorted]
        else:
            # Use PAIRAttackLLM for whitebox attack model with retry mechanism
            attack_dicts = []
            jailbreak_prompts = []
            rendered_prompts = []
            indices_to_regenerate = list(range(len(full_prompts)))

            for attempt in range(self.attack_max_attempts):
                if not indices_to_regenerate:
                    break

                if isinstance(self.attack_lm, PAIRAttackLLM):
                    batch_outputs = self.attack_lm.generate_batch(
                        [full_prompts[i] for i in indices_to_regenerate]
                    )
                else:
                    # Fallback for other model types
                    batch_outputs = self.attack_lm.generate_batch(
                        [full_prompts[i] for i in indices_to_regenerate]
                    )

                new_indices = []
                for j, full_output in enumerate(batch_outputs):
                    # All models now use text format - no processing needed
                    attack_dict, rendered_prompt = extract_json(full_output)

                    if attack_dict is None or not attack_dict.get("prompt", "").strip():
                        self.logger.warning(
                            f"Parsing failed for stream {indices_to_regenerate[j]}, will retry"
                        )
                        new_indices.append(indices_to_regenerate[j])
                    else:
                        attack_dicts.append(attack_dict)
                        # All models now use text format - use extracted prompt
                        jailbreak_prompts.append(attack_dict["prompt"])
                        rendered_prompts.append(rendered_prompt)

                indices_to_regenerate = new_indices

            # If there are still failed streams after all attempts, use fallback
            if indices_to_regenerate:
                self.logger.warning(
                    f"{len(indices_to_regenerate)} streams failed after {self.attack_max_attempts} attempts, using fallback"
                )
                for idx in indices_to_regenerate:
                    fallback_dict = {"improvement": "", "prompt": ""}
                    attack_dicts.insert(idx, fallback_dict)
                    jailbreak_prompts.insert(idx, "")
                    rendered_prompts.insert(idx, "")

        return attack_dicts, jailbreak_prompts, rendered_prompts

    def _target_batch(self, adv_prompt_list: List[str]) -> List[str]:
        """Generate responses using target model (supports both whitebox and blackbox)."""
        try:
            # Check if target model supports batch processing
            if hasattr(self.model, "generate_batch"):
                self.logger.info(
                    f"PAIR target {'blackbox' if self.target_is_blackbox else 'whitebox'}: using batch processing for {len(adv_prompt_list)} prompts"
                )
                return self.model.generate_batch(adv_prompt_list)
            else:
                # Fallback: sequential processing
                self.logger.info(
                    f"PAIR target {'blackbox' if self.target_is_blackbox else 'whitebox'}: using sequential processing for {len(adv_prompt_list)} prompts"
                )
                outputs = []
                for i, prompt in enumerate(adv_prompt_list):
                    try:
                        response = self.model.generate(prompt)
                        outputs.append(response)
                    except Exception as e:
                        self.logger.error(
                            f"PAIR target generation failed for prompt {i+1}: {e}"
                        )
                        outputs.append(f"Error: {str(e)}")
                return outputs
        except Exception as e:
            self.logger.error(f"PAIR target batch processing failed: {e}")
            # Emergency fallback: try sequential processing
            self.logger.info("PAIR attempting emergency sequential processing")
            outputs = []
            for i, prompt in enumerate(adv_prompt_list):
                try:
                    response = self.model.generate(prompt)
                    outputs.append(response)
                except Exception as seq_e:
                    self.logger.warning(
                        f"PAIR sequential fallback failed for prompt {i+1}: {seq_e}"
                    )
                    outputs.append(f"Error: {str(seq_e)}")
            return outputs

    def _judge_batch(self, adv_prompts: List[str], responses: List[str]) -> List[int]:
        return self.judge.score(adv_prompts, responses)

    def _truncate_histories(self, convs_list: List[Any]):
        truncate_histories(convs_list, self.keep_last_n)

    def generate_jailbreak(self, query: str, **kwargs) -> Dict[str, Any]:
        # Prefer dataset sample's query as goal; fallback to explicit goal if provided, then runtime default
        goal = kwargs.get("goal") or query or ""
        target_str = kwargs.get("target_str") or ""

        # Initialize conversations using attack LM template name
        convs_list, processed_response_list = self._init_conversations(
            goal, target_str, self.attack_template_name
        )

        try:
            if hasattr(self, "judge") and hasattr(self.judge, "set_context"):
                self.judge.set_context(goal, target_str)
        except Exception:
            pass

        start_time = time.time()
        best_prompt = None
        best_score = -1
        best_response = None
        best_iteration = 0

        # Create iteration progress bar
        iteration_pbar = tqdm(
            range(1, self.n_iterations + 1),
            desc="🔄 PAIR Evolution",
            unit="step",
            leave=False,
        )

        for iteration in iteration_pbar:
            extracted = self._attack_batch(
                convs_list, processed_response_list, goal, target_str
            )
            extracted_attack_list, jailbreak_prompts, rendered_prompts = extracted
            adv_prompt_list = [a["prompt"] for a in extracted_attack_list]

            # Check for empty prompts and log warnings
            for i, prompt in enumerate(adv_prompt_list):
                if not prompt.strip():
                    self.logger.warning(
                        f"Attack Prompt {i} is empty, indicating parsing failure for stream {i}"
                    )

            # Filter out empty prompts to prevent downstream errors
            valid_indices = [
                i for i, prompt in enumerate(adv_prompt_list) if prompt.strip()
            ]
            if len(valid_indices) == 0:
                self.logger.error(
                    "All attack prompts are empty, skipping this iteration"
                )
                continue

            # Use only valid prompts
            adv_prompt_list = [adv_prompt_list[i] for i in valid_indices]
            extracted_attack_list = [extracted_attack_list[i] for i in valid_indices]
            jailbreak_prompts = [jailbreak_prompts[i] for i in valid_indices]
            rendered_prompts = [rendered_prompts[i] for i in valid_indices]

            target_response_list = self._target_batch(adv_prompt_list)
            judge_scores = self._judge_batch(adv_prompt_list, target_response_list)

            # Build per-stream payload for intermediate dump
            per_stream_payload: List[Dict[str, Any]] = []
            for i in range(len(adv_prompt_list)):
                per_stream_payload.append(
                    {
                        "stream": i,
                        "candidate_json": extracted_attack_list[
                            i
                        ],  # {improvement, prompt} - parsed content
                        "rendered_prompt": rendered_prompts[
                            i
                        ],  # attacker LM raw output string
                        "jailbreak_prompt": adv_prompt_list[
                            i
                        ],  # parsed prompt sent to target (renamed from target_prompt)
                        "response": target_response_list[i],
                        "score": int(judge_scores[i]),
                        "target_str": target_str,  # target string for this sample
                        "improvement": extracted_attack_list[i].get(
                            "improvement", ""
                        ),  # improvement content
                    }
                )

            # Update progress bar with current best score
            current_best_score = max(judge_scores) if judge_scores else 0
            iteration_pbar.set_description(
                f"🔄 PAIR Evolution: Step {iteration}/{self.n_iterations} | Best: {current_best_score:.3f}"
            )

            for i, score in enumerate(judge_scores):
                if score > best_score:
                    best_score = score
                    best_prompt = adv_prompt_list[i]
                    best_response = target_response_list[i]
                    best_iteration = iteration

            # Save iteration snapshot (best_overall reflects global best so far) gated by config
            if bool(self.runtime_cfg.get("save_intermediate", False)):
                try:
                    self._save_intermediate(
                        iteration=iteration,
                        per_stream_payload=per_stream_payload,
                        best_overall={
                            "prompt": best_prompt,
                            "response": best_response,
                            "score": int(best_score),
                            "iteration": int(best_iteration),
                        },
                    )
                except Exception as e:
                    try:
                        self.logger.debug(
                            f"PAIR intermediate save error iter {iteration}: {e}"
                        )
                    except Exception:
                        pass

            # Check for early stopping
            if any(score >= self.early_stop_score for score in judge_scores):
                iteration_pbar.set_description(
                    f"🎯 SUCCESS at step {iteration} | Best: {best_score:.3f}"
                )
                break

            processed_response_list = [
                process_target_response(resp, sc, goal, target_str)
                for resp, sc in zip(target_response_list, judge_scores)
            ]

            for jailbreak_prompt, conv in zip(jailbreak_prompts, convs_list):
                conv.update_last_message(jailbreak_prompt)

            self._truncate_histories(convs_list)

        elapsed = time.time() - start_time

        print(f"\n🎯 PAIR DEBUG - Final Results ===")
        print(f"Goal: {goal}")
        print(f"Target String: {target_str}")
        print(f"Total iterations: {best_iteration}")
        print(f"Best score achieved: {best_score}")
        print(f"Early stop score threshold: {self.early_stop_score}")
        print(f"Processing time: {elapsed:.2f} seconds")
        print(
            f"Success: {best_score >= self.early_stop_score if self.early_stop_score else best_score > 0}"
        )
        print(
            f"Best prompt: {repr(best_prompt[:200]) if best_prompt else 'None'}{'...' if best_prompt and len(best_prompt) > 200 else ''}"
        )
        print(
            f"Best response: {repr(best_response[:200]) if best_response else 'None'}{'...' if best_response and len(best_response) > 200 else ''}"
        )
        # Check if best response starts with target string
        if best_response and target_str:
            starts_with_target = best_response.strip().startswith(target_str.strip())
            print(f"Best response starts with target string: {starts_with_target}")
        print("=" * 80)

        result = {
            "original_query": query,
            "jailbreak_prompt": best_prompt,
            "response": best_response,
            "success": (
                best_score >= self.early_stop_score
                if self.early_stop_score
                else best_score > 0
            ),
            "metadata": {
                "method": self.name,
                "timestamp": start_time,
                "processing_time": elapsed,
                "iterations": best_iteration,
                "best_score": int(best_score),
                "attack_type": "blackbox" if self.attack_is_blackbox else "whitebox",
            },
            "error": None,
        }
        self.update_stats(success=result["success"], error=False)
        return result

    def generate_jailbreak_batch(
        self, queries: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        batch_size = getattr(self.model, "batch_size", 1)
        # Base index for absolute dataset indexing (provided by main)
        base_index = int(kwargs.get("base_index", 0))
        dataset_key = kwargs.get("dataset_key", "unknown")
        # If attack side uses vLLM, disable dataset-level threading to avoid engine contention
        try:
            if hasattr(self, "attack_lm") and getattr(
                self.attack_lm, "use_vllm", False
            ):
                if batch_size != 1:
                    self.logger.info(
                        f"attack.use_vllm=True -> forcing dataset batch_size from {batch_size} to 1 (disable threading)"
                    )
                batch_size = 1
        except Exception:
            pass
        # Extract target_str list from kwargs
        target_str_list = kwargs.get("target_str", [])
        if not isinstance(target_str_list, list):
            target_str_list = [target_str_list] * len(queries)

        results: List[Dict[str, Any]] = []
        if batch_size <= 1:
            for i, q in enumerate(tqdm(queries, desc="PAIR", unit="q")):
                # Set thread-local per item for absolute indexing
                try:
                    self._thread_local.sample_index = base_index + i
                    self._thread_local.dataset_key = dataset_key
                except Exception:
                    pass

                # Extract individual target_str for this query
                individual_kwargs = kwargs.copy()
                individual_kwargs["target_str"] = (
                    target_str_list[i] if i < len(target_str_list) else ""
                )

                r = self.generate_jailbreak(q, **individual_kwargs)
                # Optional: attach index for downstream consumers
                try:
                    r["query_index"] = base_index + i
                except Exception:
                    pass
                results.append(r)
            return results

        self.logger.info(
            f"Processing {len(queries)} queries in dataset batches of {batch_size} with threads"
        )
        for i in range(0, len(queries), batch_size):
            sub = queries[i : i + batch_size]
            # Build query-index pairs with absolute indices for this chunk
            pairs = [
                (
                    q,
                    base_index + i + j,
                    target_str_list[i + j] if i + j < len(target_str_list) else "",
                )
                for j, q in enumerate(sub)
            ]
            with ThreadPoolExecutor(max_workers=batch_size) as ex:

                def _run_one(q_with_idx_and_target):
                    q, abs_idx, target_str = q_with_idx_and_target
                    try:
                        self._thread_local.sample_index = abs_idx
                        self._thread_local.dataset_key = dataset_key
                    except Exception:
                        pass

                    # Extract individual target_str for this query
                    individual_kwargs = kwargs.copy()
                    individual_kwargs["target_str"] = target_str

                    r = self.generate_jailbreak(q, **individual_kwargs)
                    try:
                        r["query_index"] = abs_idx
                    except Exception:
                        pass
                    return r

                future_map = {ex.submit(_run_one, p): p for p in pairs}
                for fut in as_completed(future_map):
                    results.append(fut.result())
        return results

    def prepare_prompt(self, query: str, **kwargs) -> str:
        # PAIR does not transform the input query at this stage; it uses dataset query as goal.
        return (query or "").strip()
