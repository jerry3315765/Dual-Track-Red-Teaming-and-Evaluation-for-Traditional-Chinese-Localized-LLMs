import argparse
import json
import os
import time
import gc
from typing import Dict, List, Tuple

try:
    import torch
except ImportError:
    torch = None

from integrate_redteam_divi import load_config
from src.models.local_llm import LocalLLM

ERROR_PATTERNS = [
    "LM Studio Error: HTTPConnectionPool(host='localhost', port=1234): Read timed out. (read timeout=120)",
    "API Error: 400 - {\"error\":\"Error rendering prompt with jinja template: \\\"Only user and assistant roles are supported!\\\"",
]


def is_bad_response(text: str) -> bool:
    if not text:
        return False
    return any(p in text for p in ERROR_PATTERNS)


def load_results_files(output_dir: str) -> List[str]:
    return [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("redteam_divi_results_") and f.endswith(".json")
    ]


def build_model_messages(system_prompt: str, history_records: List[dict], use_system_prompt: bool) -> List[dict]:
    messages = []
    if system_prompt:
        if use_system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "user", "content": f"[System Prompt]\n{system_prompt}"})
    for r in history_records:
        messages.append({"role": "user", "content": r["prompt"]})
        messages.append({"role": "assistant", "content": r["response"]})
    return messages


def group_records(data: List[dict]) -> Dict[Tuple, Dict[int, dict]]:
    groups: Dict[Tuple, Dict[int, dict]] = {}
    for r in data:
        key = (r.get("scenario_id"), r.get("model"), r.get("system_prompt_name"))
        if key not in groups:
            groups[key] = {}
        groups[key][int(r.get("turn", 0))] = r
    return groups


def retry_group(
    records_by_turn: Dict[int, dict],
    llm: LocalLLM,
    model_name: str,
    max_retries: int,
    sleep_sec: int,
) -> int:
    turns = sorted(records_by_turn.keys())
    if not turns:
        return 0

    bad_turns = [t for t in turns if is_bad_response(records_by_turn[t].get("response", ""))]
    if not bad_turns:
        return 0

    start_turn = min(bad_turns)
    max_turn = max(turns)

    system_prompt = records_by_turn[turns[0]].get("system_prompt", "")
    # Build history up to start_turn - 1
    history_records = []
    for t in turns:
        if t >= start_turn:
            break
        history_records.append({
            "turn": t,
            "prompt": records_by_turn[t].get("prompt", ""),
            "response": records_by_turn[t].get("response", ""),
        })

    use_system_prompt = "mistral-7b-instruct-v0.3" not in (model_name or "")
    model_messages = build_model_messages(system_prompt, history_records, use_system_prompt)
    conversation_history = history_records.copy()

    fixed_count = 0

    for t in range(start_turn, max_turn + 1):
        if t not in records_by_turn:
            continue

        user_prompt = records_by_turn[t].get("prompt", "")
        model_messages.append({"role": "user", "content": user_prompt})

        new_response = None
        for attempt in range(max_retries):
            response_result = llm.generate([
                {
                    "prompt": model_messages,
                    "template_id": f"retry_scenario_{records_by_turn[t].get('scenario_id')}_turn_{t}_attempt_{attempt + 1}",
                    "intent": "red_team_attack",
                }
            ])[0]
            new_response = response_result.get("response", "")
            if new_response and not is_bad_response(new_response):
                break
            time.sleep(sleep_sec)

        if not new_response:
            new_response = ""

        model_messages.append({"role": "assistant", "content": new_response})
        conversation_history.append({"turn": t, "prompt": user_prompt, "response": new_response})

        record = records_by_turn[t]
        record["response"] = new_response
        record["conversation_history"] = conversation_history.copy()

        fixed_count += 1

    return fixed_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry bad samples in redteam results")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "results"),
        help="Directory containing redteam_divi_results_*.json",
    )
    parser.add_argument(
        "--models-config",
        default=os.path.join(os.path.dirname(__file__), "config", "models.yaml"),
        help="Path to models.yaml",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=int, default=5)
    args = parser.parse_args()

    models_config = load_config(args.models_config)
    model_map = {m["name"]: m for m in models_config["models"]}
    # Handle deepseek alias if present
    deepseek_alias = "lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf"
    if "deepseek-r1-distill-llama-8b@q8_0" in model_map:
        model_map[deepseek_alias] = model_map["deepseek-r1-distill-llama-8b@q8_0"]
    if "deepseek-r1-distill-llama-8b@Q8_0" in model_map:
        model_map[deepseek_alias] = model_map["deepseek-r1-distill-llama-8b@Q8_0"]

    files = load_results_files(args.output_dir)
    if not files:
        print("No result files found.")
        return

    # Phase 1: Scan all files to build a work plan
    print("Scanning files to build work plan...")
    # Structure: model_name -> dict[file_path -> list of group_keys]
    work_plan: Dict[str, Dict[str, List[Tuple]]] = {}
    
    total_files = len(files)
    for i, file_path in enumerate(files):
        print(f"  Scanning [{i+1}/{total_files}] {os.path.basename(file_path)}... ", end="", flush=True)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load: {e}")
            continue

        groups = group_records(data)
        found_issues = 0
        for key, records_by_turn in groups.items():
            # key = (scenario_id, model_name, system_prompt_name)
            model_name = key[1]
            
            # Check for bad responses
            turns = sorted(records_by_turn.keys())
            if any(is_bad_response(records_by_turn[t].get("response", "")) for t in turns):
                if model_name not in work_plan:
                    work_plan[model_name] = {}
                if file_path not in work_plan[model_name]:
                    work_plan[model_name][file_path] = []
                work_plan[model_name][file_path].append(key)
                found_issues += 1
        
        print(f"Found {found_issues} groups to retry.")
        
        # Explicitly free memory
        del data
        del groups
        gc.collect()

    sorted_models = sorted(work_plan.keys())
    print(f"\nWork plan ready. Models to process: {', '.join(sorted_models)}")

    current_llm: LocalLLM = None
    total_fixed_all_sessions = 0
    SAVE_INTERVAL = 5

    # Phase 2: Process by Model
    for model_name in sorted_models:
        file_tasks = work_plan[model_name]
        if not file_tasks:
            continue
        
        print(f"\n=== Processing Model: {model_name} ===")
        
        # 1. Unload previous model & Load new Model
        if current_llm:
            del current_llm
            current_llm = None
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_cfg = model_map.get(model_name)
        if not model_cfg or model_cfg.get("type") != "local":
            print(f"  Skip tasks for {model_name} (no local config found)")
            continue

        print(f"  Loading model...")
        try:
            current_llm = LocalLLM(model_cfg)
        except Exception as e:
            print(f"  Failed to load model {model_name}: {e}")
            continue

        # 2. Process all files that need this model
        # Sort files to ensure deterministic order
        sorted_files = sorted(file_tasks.keys())
        
        for file_path in sorted_files:
            group_keys = file_tasks[file_path]
            fn = os.path.basename(file_path)
            print(f"  > File: {fn} ({len(group_keys)} scenarios)")
            
            # Reload file content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"    Error reading file: {e}")
                continue
            
            groups = group_records(data)
            
            file_updates = 0
            for key in group_keys:
                if key not in groups:
                    continue  # Should not happen if file hasn't explicitly changed structure

                records_by_turn = groups[key]
                fixed = retry_group(
                    records_by_turn,
                    current_llm, 
                    model_name,
                    args.max_retries, 
                    args.sleep_sec
                )
                
                if fixed > 0:
                    file_updates += fixed
                    total_fixed_all_sessions += fixed
                    
                    # Incremental save within the file processing
                    if file_updates % SAVE_INTERVAL == 0:
                        try:
                            # print(f"    Saving progress... ({file_updates} fixes)")
                            with open(file_path, "w", encoding="utf-8") as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            print(f"    Error saving progress: {e}")

            if file_updates > 0:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    print(f"    Done. Fixed {file_updates} samples in {fn}.")
                except Exception as e:
                    print(f"    Error saving final file content: {e}")
            else:
                print(f"    No changes made to {fn} (maybe retries failed).")

            # Clean memory for file data immediately
            del data
            del groups
            gc.collect()

    print(f"\nAll done. Total fixed records: {total_fixed_all_sessions}")


if __name__ == "__main__":
    main()
