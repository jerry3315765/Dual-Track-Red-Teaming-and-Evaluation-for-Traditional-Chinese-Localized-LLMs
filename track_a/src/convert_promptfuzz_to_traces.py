import json
import os
import pandas as pd
import argparse
import ast

from src.evaluation.re_evaluate_results import evaluate_turn, first_nonempty_text


def safe_parse_list(value, fallback):
    if isinstance(value, list):
        return value
    if value is None:
        return fallback
    if pd.isna(value):
        return fallback
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return fallback
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return fallback
    return fallback


def resolve_root_prompt(index_to_row, row_index, cache):
    if row_index in cache:
        return cache[row_index]

    row = index_to_row.get(row_index)
    if row is None:
        cache[row_index] = ""
        return ""

    parent_value = row.get("parent")
    try:
        parent_index = int(parent_value)
    except (TypeError, ValueError):
        parent_index = None

    current_prompt = first_nonempty_text(row, ("prompt", "attack", "mutated_prompt", "query"), default="")
    if parent_index is None or parent_index not in index_to_row or parent_index == row_index:
        cache[row_index] = current_prompt
        return current_prompt

    root_prompt = resolve_root_prompt(index_to_row, parent_index, cache)
    cache[row_index] = root_prompt or current_prompt
    return cache[row_index]

def convert_promptfuzz_to_traces(results_csv, output_json, model_name="gemma-3-taide-12b-chat"):
    """
    Reads PromptFuzz output (from Results/focus/redteam/*.csv)
    and converts to thesis-experiment raw_traces format.
    """
    if not os.path.exists(results_csv):
        print(f"File not found: {results_csv}. Please run PromptFuzz first.")
        return

    df = pd.read_csv(results_csv)
    if "index" in df.columns:
        df["index"] = pd.to_numeric(df["index"], errors="coerce")
    if "parent" in df.columns:
        df["parent"] = pd.to_numeric(df["parent"], errors="coerce")

    index_to_row = {}
    for _, row in df.iterrows():
        row_index = row.get("index")
        if pd.notna(row_index):
            index_to_row[int(row_index)] = row

    root_prompt_cache = {}
    
    # Load Defense Mapping if available
    mapping_path = os.path.join(os.path.dirname(results_csv), "../../../../Datasets/redteam_focus_defense_mapping.json")
    # Resolve relative path
    mapping_path = os.path.abspath(mapping_path)
    
    defense_map = {}
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                defense_list = json.load(f)
                for d in defense_list:
                    defense_map[d['id']] = d['name']
        except:
            print("Warning: Could not load defense mapping.")

    traces = []
    for idx, row in df.iterrows():
        # PromptFuzz output columns: 'prompt', 'response', 'results'
        # 'response' and 'results' are string representation of lists
        mutated_prompt = first_nonempty_text(row, ("prompt", "attack", "mutated_prompt", "query"), default="")
        row_index = int(row["index"]) if pd.notna(row.get("index")) else idx
        original_prompt = resolve_root_prompt(index_to_row, row_index, root_prompt_cache)
        if not original_prompt:
            original_prompt = first_nonempty_text(row, ("attack", "parent_prompt"), default=mutated_prompt)
        
        responses = safe_parse_list(row.get("response"), [str(row.get("response", ""))])
        results = safe_parse_list(row.get("results"), [0])
        if len(results) < len(responses):
            results = results + [0] * (len(responses) - len(results))

        # Each prompt is tested against multiple defenses (scenarios).
        # We need to map them back if possible, or treat them as individual samples.
        # Since we don't have the explicit defense ID in the CSV row easily (it's implicit in the order),
        # We will generate one trace per response.
        
        for i, (resp, res) in enumerate(zip(responses, results)):
            defense_name = defense_map.get(i, f"Defense_{i}")
            scenario_desc = f"PromptFuzz Mutated Prompt ({defense_name})"

            judge_input = {
                "prompt": mutated_prompt,
                "response": resp,
                "original_prompt": original_prompt,
                "scenario_desc": scenario_desc,
            }
            try:
                eval_result = evaluate_turn(judge_input)
            except Exception as exc:
                eval_result = {
                    "score": 0,
                    "success": bool(res),
                    "classification": "Error",
                    "prompt_intent": "unclear",
                    "attack_valid": False,
                    "response_harmfulness": "unclear",
                    "reason": f"Judge evaluation failed: {exc}",
                }

            is_success = bool(eval_result.get("success", False))
            
            trace = {
                "scenario_id": idx, # Using row index as proxy for ID group
                "scenario_desc": scenario_desc,
                "attack_type": "promptfuzz",
                "system_prompt_name": defense_name,
                "system_prompt": "See defense mapping",
                "turn": 1,
                "prompt": mutated_prompt,
                "original_prompt": original_prompt,
                "parent_prompt": first_nonempty_text(row, ("parent_prompt",), default=""),
                "mutated_prompt": mutated_prompt,
                "response": resp,
                "evaluation": eval_result,
                "model": model_name,
                "success": is_success,
                "conversation_history": [
                    {"role": "user", "content": mutated_prompt},
                    {"role": "assistant", "content": resp}
                ]
            }
            traces.append(trace)

    # Make sure output dir exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
        
    print(f"Converted {len(traces)} PrompFuzz results -> {output_json}")
    print("Next step: You can run src/integrate_redteam_divi.py or analyze_success_rates.py using this JSON.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="PromptFuzz-Thesis/Results/focus/redteam/0.csv")
    parser.add_argument("--out", type=str, default="thesis-experiment/results/raw_traces/redteam_divi_results_promptfuzz.json")
    parser.add_argument("--model", type=str, default="gemma-3-taide-12b-chat")
    args = parser.parse_args()
    
    convert_promptfuzz_to_traces(args.csv, args.out, args.model)
