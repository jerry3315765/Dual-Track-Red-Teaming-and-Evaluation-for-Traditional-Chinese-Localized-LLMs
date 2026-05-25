import os
import sys
import json
import glob
import pandas as pd
from tqdm import tqdm

# --- Setup Paths & Environment ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
workspace_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.evaluation.judge import Judge

candidate_results_dirs = [
    os.path.join(project_root, "results", "raw_traces"),
    os.path.join(project_root, "results", "exported"),
    os.path.join(project_root, "experiment_v2", "data", "results"),
]

env_path = os.path.join(project_root, ".env")

# --- Load Environment Variables ---
def load_env(path):
    if not os.path.exists(path):
        print(f"Warning: .env file not found at {path}")
        return
    
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip("'").strip('"')

load_env(env_path)

judge = Judge({"judge_model": "gpt-4o-mini"})

# --- Configuration ---
EVAL_MODEL = "gpt-4o-mini"

PROMPT_FIELD_CANDIDATES = (
    "prompt",
    "mutated_prompt",
    "attack",
    "current_prompt",
    "user_prompt",
    "input",
    "query",
)

RESPONSE_FIELD_CANDIDATES = (
    "response",
    "assistant_response",
    "output",
    "generated_response",
)

ORIGINAL_PROMPT_FIELD_CANDIDATES = (
    "original_prompt",
    "seed_prompt",
    "attack_prompt",
    "source_prompt",
    "base_prompt",
    "parent_prompt",
    "root_prompt",
)


def first_nonempty_text(item, field_names, default=""):
    for key in field_names:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is None:
            continue
        if pd.isna(value):
            continue
        if value is not None and not isinstance(value, (dict, list, tuple, set)):
            text = str(value).strip()
            if text:
                return text
    return default

def find_result_files():
    files = []
    seen = set()

    for directory in candidate_results_dirs:
        if not os.path.isdir(directory):
            continue

        pattern = os.path.join(directory, "redteam_divi_results_*.json")
        for file_path in sorted(glob.glob(pattern)):
            if file_path in seen:
                continue
            seen.add(file_path)
            files.append(file_path)

    return files


def extract_original_prompt(item):
    return first_nonempty_text(item, ORIGINAL_PROMPT_FIELD_CANDIDATES, default=None)


def extract_current_prompt(item):
    return first_nonempty_text(item, PROMPT_FIELD_CANDIDATES, default="")


def extract_response_text(item):
    return first_nonempty_text(item, RESPONSE_FIELD_CANDIDATES, default="")


def evaluate_turn(item):
    """Calls the shared judge with optional original attack context."""
    scenario = item.get("original_scenario_desc") or item.get("scenario_desc") or "Unknown Scenario"
    prompt = extract_current_prompt(item)
    response = extract_response_text(item)
    original_prompt = extract_original_prompt(item)

    try:
        return judge.evaluate(
            prompt=prompt,
            response=response,
            original_prompt=original_prompt,
            scenario=scenario,
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {
            "score": 0,
            "success": False,
            "classification": "Error",
            "prompt_intent": "unclear",
            "attack_valid": False,
            "response_harmfulness": "unclear",
            "reason": f"API Error: {str(e)}",
        }

def process_file(file_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON {file_path}: {e}")
        return

    updated_count = 0

    # Use tqdm to show progress bar for items in the file
    for item in tqdm(data, desc="Evaluating", unit="turn"):
        # Extract necessary fields
        response_text = extract_response_text(item)

        # Skip if response is empty
        if not response_text:
            continue

        # Perform Evaluation
        eval_result = evaluate_turn(item)

        # Update Item
        item["evaluation"] = eval_result
        item["success"] = bool(eval_result.get("success", False)) # Overwrite success based on LLM judge
        item["prompt_intent"] = eval_result.get("prompt_intent")
        item["attack_valid"] = eval_result.get("attack_valid")
        item["response_harmfulness"] = eval_result.get("response_harmfulness")

        updated_count += 1

        # Optional: Rate limit protection?
        # time.sleep(0.1) 

    # Save updated JSON
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Updated {file_path} with {updated_count} evaluations.")
    except Exception as e:
        print(f"Error saving JSON {file_path}: {e}")
        return

    # Update/Regenerate CSV
    try:
        # Create CSV filename based on JSON filename
        csv_path = file_path.replace(".json", ".csv")
        # Flatten the data for CSV
        # We need to flatten the 'evaluation' nested dict
        flat_data = []
        for d in data:
            flat_item = d.copy()
            # Flatten evaluation
            eval_data = flat_item.pop("evaluation", {})
            flat_item["eval_score"] = eval_data.get("score")
            flat_item["eval_success"] = eval_data.get("success")
            flat_item["eval_classification"] = eval_data.get("classification")
            flat_item["eval_prompt_intent"] = eval_data.get("prompt_intent")
            flat_item["eval_attack_valid"] = eval_data.get("attack_valid")
            flat_item["eval_response_harmfulness"] = eval_data.get("response_harmfulness")
            flat_item["eval_reason"] = eval_data.get("reason")

            # Remove complex nested structures like conversation_history for CSV readability if desired,
            # but usually we keep them or stringify them. 
            # For now, let's keep them as pandas handles them or we can drop them if too large.
            # Usually strict CSVs don't like lists.
            if "conversation_history" in flat_item:
                flat_item["conversation_history"] = json.dumps(flat_item["conversation_history"], ensure_ascii=False)
            
            flat_data.append(flat_item)

        df = pd.DataFrame(flat_data)
        # Reorder columns to put evaluation results near specific columns if needed
        # Just save it
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Regenerated CSV: {os.path.basename(csv_path)}")

    except Exception as e:
        print(f"Error generating CSV for {file_path}: {e}")

def main():
    files = find_result_files()

    if not files:
        print("No result files found in the known results directories.")
        return

    print(f"Found {len(files)} files to evaluate.")

    for file in files:
        process_file(file)

if __name__ == "__main__":
    main()
