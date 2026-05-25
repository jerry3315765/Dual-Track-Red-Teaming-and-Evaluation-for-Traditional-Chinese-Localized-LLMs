import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)
TRACK_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
BUNDLE_ROOT = os.path.abspath(os.path.join(TRACK_ROOT, '..'))

# Ensure src is in python path
sys.path.append(BASE_DIR)

from core.models.local_llm import LocalLLM
from core.evaluation.judge import Judge

def load_yaml(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return yaml.safe_load(f)

def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def safe_name(value):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in str(value))

def build_conversations_from_scenarios(scenarios):
    conversations = []
    for scenario in scenarios:
        scenario_id = scenario.get('scenario_id') or scenario.get('id')
        steps = list(scenario.get('steps') or scenario.get('prompts') or [])
        if not steps:
            continue
        conversations.append({
            "intent": scenario.get('description', scenario.get('title', '')),
            "scenario_id": scenario_id,
            "scenario_domain": scenario.get('domain', ''),
            "scenario_persona": scenario.get('persona', ''),
            "scenario_description": scenario.get('description', ''),
            "scenario_context": scenario.get('context', ''),
            "scenario_constraints": scenario.get('constraints', ''),
            "target_harmful_output": scenario.get('target_harmful_output', ''),
            "example_question": scenario.get('example_question', ''),
            "attack_method": scenario.get('description', ''),
            "steps": [str(step).strip() for step in steps],
        })
    return conversations


def load_system_prompts(track_root: str) -> Dict[str, str]:
    prompt_dir = os.path.join(track_root, "data", "raw", "attackSystemPromptChinese")
    prompts: Dict[str, str] = {}
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"System prompt directory not found: {prompt_dir}")

    for name in sorted(os.listdir(prompt_dir)):
        if not name.endswith(".txt"):
            continue
        key = name[:-4]
        with open(os.path.join(prompt_dir, name), "r", encoding="utf-8") as f:
            prompts[key] = f.read().strip()

    if not prompts:
        raise ValueError(f"No system prompt files found under: {prompt_dir}")
    return prompts

def save_results(results, output_dir=None, name=None, exact_path=None):
    if exact_path:
        filepath = exact_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{name}_{timestamp}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {filepath}")
    return filepath


def generate_model_results(model_cfg, conversations, system_prompts, base_url):
    model_results = []
    provider = model_cfg.get('provider') or model_cfg.get('type')
    if provider not in {'local', 'openai-compatible', 'remote'}:
        raise ValueError(f"Unsupported provider for {model_cfg['name']}: {provider}")

    resolved_base_url = base_url or model_cfg.get('base_url') or model_cfg.get('api_base')
    if not resolved_base_url:
        raise ValueError(f"Model {model_cfg['name']} is missing base_url/api_base and no CLI override was provided")

    llm = LocalLLM({
        "name": model_cfg.get('name'),
        "path": model_cfg.get('model_path') or model_cfg.get('path'),
        "quantization": model_cfg.get('quantization', 'none'),
        "api_base": resolved_base_url,
        "served_model_name": model_cfg.get('served_model_name') or model_cfg.get('name'),
        "api_key_env": model_cfg.get('api_key_env', 'OPENAI_API_KEY'),
        "device": model_cfg.get('device'),
    })

    for prompt_name, system_prompt in system_prompts.items():
        print(f"Running system prompt: {prompt_name} ({len(conversations)} conversations)")
        for conv in conversations:
            history = []
            steps = conv.get("steps", [])
            for idx, step in enumerate(steps):
                row = {
                    "template_id": f"{conv.get('scenario_id')}-{idx}",
                    "intent": conv.get("intent", ""),
                    "scenario_id": conv.get("scenario_id"),
                    "scenario_domain": conv.get("scenario_domain", ""),
                    "scenario_persona": conv.get("scenario_persona", ""),
                    "scenario_description": conv.get("scenario_description", ""),
                    "scenario_context": conv.get("scenario_context", ""),
                    "scenario_constraints": conv.get("scenario_constraints", ""),
                    "target_harmful_output": conv.get("target_harmful_output", ""),
                    "example_question": conv.get("example_question", ""),
                    "attack_method": conv.get("attack_method", ""),
                    "turn_index": idx + 1,
                    "turn_total": len(steps),
                    "prompt": step,
                    "history": history,
                    "system_prompt_name": prompt_name,
                    "system_prompt": system_prompt,
                }
                turn_result = llm.generate([row])[0]
                model_results.append(turn_result)
                history = [
                    *history,
                    {"role": "user", "content": step},
                    {"role": "assistant", "content": turn_result.get("response", "")},
                ]

    del llm
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return model_results


def evaluate_results(results, judge):
    for res in results:
        eval_result = judge.evaluate(
            res['prompt'],
            res['response'],
            original_prompt=res.get('intent'),
            scenario=res.get('scenario_description') or res.get('scenario_id'),
        )
        res['evaluation'] = eval_result
    return results

def main():
    print("Starting Traditional Chinese LLM Jailbreak Experiment V2...")

    # Keep runtime OPENAI_API_KEY from parent process; treat .env as fallback only.
    load_dotenv(os.path.join(BUNDLE_ROOT, ".env"), override=False)

    parser = argparse.ArgumentParser(description="Track A runner")
    parser.add_argument("--models", default=os.path.join(BUNDLE_ROOT, "models.yaml"))
    parser.add_argument("--scenarios", default=os.path.join(TRACK_ROOT, "config", "red_team_scenarios copy.json"))
    parser.add_argument("--model_name", default=None, help="Run a single model by name")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_label", default=None, help="Optional label used for output folders")
    parser.add_argument("--base_url", default=None, help="Override model base_url with an OpenAI-compatible endpoint")
    parser.add_argument("--mode", choices=["full", "generate", "evaluate"], default="full")
    parser.add_argument("--input_results_path", default=None, help="Input JSON path for evaluate mode")
    parser.add_argument("--raw_results_path", default=None, help="Exact output JSON path for generate mode")
    parser.add_argument("--evaluated_results_path", default=None, help="Exact output JSON path for evaluate/full mode")
    args = parser.parse_args()

    # 1. Load Configurations
    config = load_yaml(os.path.join(TRACK_ROOT, 'config', 'config.yaml'))
    models_config = load_yaml(args.models)
    conversations = None
    system_prompts = None
    if args.mode != "evaluate":
        scenarios_data = load_json(args.scenarios)
        system_prompts = load_system_prompts(TRACK_ROOT)

        print("Generating attack prompts...")
        scenarios = scenarios_data.get('scenarios', [])
        conversations = build_conversations_from_scenarios(scenarios)
        if not conversations:
            raise ValueError("No scenarios were loaded from the configured scenarios file")
        total_turns = sum(len(conv.get("steps", [])) for conv in conversations)
        print(f"Loaded {len(conversations)} conversation scenarios, {total_turns} total turns.")
        print(f"Loaded {len(system_prompts)} system prompts.")

    output_dir = args.output_dir or os.path.join(TRACK_ROOT, config.get('output_dir', 'data/results'))
    os.makedirs(output_dir, exist_ok=True)

    judge = Judge(config)
    all_results = []
    
    for model_cfg in models_config['models']:
        if args.model_name and model_cfg.get('name') != args.model_name:
            continue
        print(f"Processing model: {model_cfg['name']}")

        model_output_dir = os.path.join(output_dir, safe_name(args.run_label or model_cfg['name']))
        os.makedirs(model_output_dir, exist_ok=True)

        if args.mode == "evaluate":
            if not args.input_results_path:
                raise ValueError("--input_results_path is required in evaluate mode")
            with open(args.input_results_path, "r", encoding="utf-8") as f:
                model_results = json.load(f)
        else:
            model_results = generate_model_results(
                model_cfg=model_cfg,
                conversations=conversations,
                system_prompts=system_prompts,
                base_url=args.base_url,
            )

        if not model_results:
            raise RuntimeError(f"Model {model_cfg['name']} returned no results")

        if args.mode in {"full", "evaluate"}:
            print(f"Evaluating results for {model_cfg['name']}...")
            model_results = evaluate_results(model_results, judge)

        run_key = safe_name(args.run_label or model_cfg['name'])
        if args.mode == "generate":
            save_results(
                model_results,
                exact_path=args.raw_results_path,
                output_dir=model_output_dir,
                name=f"raw_results_{run_key}",
            )
        elif args.mode == "evaluate":
            save_results(
                model_results,
                exact_path=args.evaluated_results_path,
                output_dir=model_output_dir,
                name=f"results_{run_key}",
            )
        else:
            save_results(
                model_results,
                exact_path=args.evaluated_results_path,
                output_dir=model_output_dir,
                name=f"results_{run_key}",
            )
        all_results.extend(model_results)

    # Save aggregated results
    if all_results:
        if args.mode != "generate":
            save_results(all_results, output_dir=output_dir, name="final_results_aggregated")
        print("Experiment run complete.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
