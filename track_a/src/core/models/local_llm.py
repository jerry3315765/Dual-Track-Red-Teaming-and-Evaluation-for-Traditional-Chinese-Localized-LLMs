import json
import os
import re

import requests
from openai import OpenAI
from tqdm import tqdm

try:
    import torch
except Exception:
    torch = None

class LocalLLM:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('path')
        self.api_base = config.get('api_base') or config.get('base_url')
        self.api_key = config.get('api_key') or os.getenv(config.get('api_key_env', 'OPENAI_API_KEY'), '')
        self.served_model_name = config.get('served_model_name') or config.get('name') or self.model_path

        if not self.api_base:
            raise ValueError(f"api_base is required for {self.config.get('name')} and local fallback has been removed")

        print(f"Loading model: {self.config.get('name')} from {self.model_path}")
        self._load_model()

    def _load_model(self):
        try:
            self.use_remote_api = True
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            response = requests.get(f"{self.api_base}/models", timeout=10)
            if response.status_code != 200:
                raise RuntimeError(f"API not ready: {response.status_code}")
            print("Remote OpenAI-compatible API connection successful")
        except Exception as e:
            print(f"Error loading model {self.config.get('name')}: {e}")
            raise e

    def generate(self, prompts):
        results = []
        for item in tqdm(prompts, desc=f"Generating with {self.config.get('name')}"):
            # Handle prompt as string or list of messages
            prompt_input = item['prompt']
            
            # Prepare messages list for chat models
            if isinstance(prompt_input, str):
                history = item.get("history") or []
                system_prompt = item.get("system_prompt")
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if history:
                    messages.extend(history)
                messages.append({"role": "user", "content": prompt_input})
            else:
                messages = prompt_input
            
            if self.use_remote_api:
                # Use remote OpenAI-compatible API (vLLM)
                target_max_tokens = 4096
                response_text = None
                for _ in range(3):
                    try:
                        completion = self.client.chat.completions.create(
                            model=self.served_model_name,
                            messages=messages,
                            max_tokens=target_max_tokens,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        response_text = completion.choices[0].message.content
                        break
                    except Exception as e:
                        msg = str(e)
                        lowered = msg.lower()
                        # Handle vLLM context-length validation gracefully:
                        # shrink max_tokens and retry instead of failing the turn.
                        if ("maximum context length" in lowered or "requested" in lowered) and ("input token" in lowered):
                            max_ctx_match = re.search(r"maximum context length is\s+(\d+)", msg, re.IGNORECASE)
                            input_tok_match = re.search(r"prompt contains at least\s+(\d+)\s+input tokens", msg, re.IGNORECASE)
                            requested_match = re.search(r"requested\s+(\d+)\s+output tokens", msg, re.IGNORECASE)
                            max_ctx = int(max_ctx_match.group(1)) if max_ctx_match else 8000
                            input_toks = int(input_tok_match.group(1)) if input_tok_match else None
                            requested = int(requested_match.group(1)) if requested_match else target_max_tokens
                            if input_toks is not None:
                                adjusted = max(256, min(requested - 1, max_ctx - input_toks - 8))
                            else:
                                adjusted = max(256, min(requested - 1, target_max_tokens - 512))
                            if adjusted < target_max_tokens:
                                target_max_tokens = adjusted
                                continue
                        response_text = f"Remote API Error: {msg}"
                        break
                if response_text is None:
                    response_text = "Remote API Error: exhausted retries without response"

            else:
                if torch is None:
                    raise RuntimeError("Local generation backend requires torch, but torch is not installed.")
                # Original transformers generation
                try:
                    input_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    # Fallback if no chat template or raw string
                    if isinstance(prompt_input, str):
                        input_text = prompt_input
                    else:
                         # Simple join if list but no template
                         input_text = "\n".join([m['content'] for m in messages])

                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )

                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Store original prompt/messages in result for reference
            results.append({
                "model": self.config.get('name'),
                "template_id": item['template_id'],
                "intent": item['intent'],
                "system_prompt_name": item.get("system_prompt_name"),
                "scenario_id": item.get("scenario_id"),
                "scenario_domain": item.get("scenario_domain"),
                "scenario_persona": item.get("scenario_persona"),
                "scenario_description": item.get("scenario_description"),
                "scenario_context": item.get("scenario_context"),
                "scenario_constraints": item.get("scenario_constraints"),
                "target_harmful_output": item.get("target_harmful_output"),
                "example_question": item.get("example_question"),
                "attack_method": item.get("attack_method"),
                "turn_index": item.get("turn_index"),
                "turn_total": item.get("turn_total"),
                "prompt": prompt_input if isinstance(prompt_input, str) else json.dumps(prompt_input, ensure_ascii=False),
                "response": response_text
            })

        return results
