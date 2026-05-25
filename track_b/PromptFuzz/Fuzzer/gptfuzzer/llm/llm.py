from openai import OpenAI
import logging
import time
import concurrent.futures
import re



class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 base_url=None,
                 system_message=None
                ):
        super().__init__()

        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        self.model_path = model_path
        self.post_prompt = None
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    @staticmethod
    def _estimate_prompt_tokens(messages):
        # Conservative heuristic for mixed Chinese/English prompts.
        total_chars = 0
        for msg in messages:
            total_chars += len(str(msg.get("content", "")))
        return max(1, total_chars)

    @staticmethod
    def _extract_context_limit(error_text):
        match = re.search(r"maximum context length is\s+(\d+)\s+tokens", error_text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_input_tokens(error_text):
        match = re.search(r"prompt contains at least\s+(\d+)\s+input tokens", error_text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def generate(self, prompt, temperature=0, max_tokens=4096, n=1, max_trials=3, failure_sleep_time=2, target=None):
        if target is None:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
        else:
            self.system_message = target['pre_prompt']
            self.post_prompt = target['post_prompt']
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            # Note: post_prompt is effectively ignored in typical chat structure unless appended to user content
            if self.post_prompt:
                messages.append({"role": "user", "content": self.post_prompt})
                
        # Handle cases where messages > 1 (multi-turn simulation or system prompt injection)
        # Track B is technically Single Turn (System + User), but the Fuzzer engine here creates one message block.
        # This confirms it's a single exchange from the model's perspective.

        requested_max_tokens = max_tokens
        for trial in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=requested_max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                error_text = str(e)
                context_limit = self._extract_context_limit(error_text)
                if context_limit is not None:
                    prompt_tokens = self._extract_input_tokens(error_text)
                    if prompt_tokens is None:
                        prompt_tokens = self._estimate_prompt_tokens(messages)
                    # Leave buffer to avoid edge failures from tokenizer differences.
                    adjusted = max(32, context_limit - prompt_tokens - 128)
                    if adjusted < requested_max_tokens:
                        logging.warning(
                            "Context-length error detected. Auto-reducing max_tokens "
                            f"from {requested_max_tokens} to {adjusted} and retrying immediately."
                        )
                        requested_max_tokens = adjusted
                        continue
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {trial + 1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=4096, n=1, max_trials=3, failure_sleep_time=2, target=None):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time, target): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class OpenAIEmbeddingLLM():
    def __init__(self, model_path=None, api_key=None):
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
    
    def get_embedding(self, prompt):
        response = self.client.embeddings.create(
            input=prompt,
            model=self.model_path
        )
        
        return response.data[0].embedding
