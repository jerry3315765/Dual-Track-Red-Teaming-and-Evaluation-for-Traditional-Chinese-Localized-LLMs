import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
TRACK_ROOT = BASE_DIR.parent
BUNDLE_ROOT = TRACK_ROOT.parent

sys.path.append(str(BASE_DIR))

from core.evaluation.judge import Judge
from core.models.local_llm import LocalLLM


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))


def build_conversations_from_scenarios(scenarios: List[Dict]) -> List[Dict]:
    conversations: List[Dict] = []
    for scenario in scenarios:
        scenario_id = scenario.get("scenario_id") or scenario.get("id")
        steps = list(scenario.get("steps") or scenario.get("prompts") or [])
        if not steps:
            continue
        conversations.append(
            {
                "intent": scenario.get("description", scenario.get("title", "")),
                "scenario_id": scenario_id,
                "scenario_domain": scenario.get("domain", ""),
                "scenario_persona": scenario.get("persona", ""),
                "scenario_description": scenario.get("description", ""),
                "scenario_context": scenario.get("context", ""),
                "scenario_constraints": scenario.get("constraints", ""),
                "target_harmful_output": scenario.get("target_harmful_output", ""),
                "example_question": scenario.get("example_question", ""),
                "attack_method": scenario.get("description", ""),
                "steps": [str(step).strip() for step in steps],
            }
        )
    return conversations


def load_system_prompts(track_root: Path) -> Dict[str, str]:
    prompt_dir = track_root / "data" / "raw" / "attackSystemPromptChinese"
    if not prompt_dir.is_dir():
        raise FileNotFoundError(f"System prompt directory not found: {prompt_dir}")

    prompts: Dict[str, str] = {}
    for entry in sorted(prompt_dir.iterdir()):
        if entry.is_file() and entry.suffix == ".txt":
            prompts[entry.stem] = entry.read_text(encoding="utf-8").strip()

    if not prompts:
        raise ValueError(f"No system prompt files found under: {prompt_dir}")
    return prompts


def save_json(data: List[Dict], output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{name}_{stamp}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results to {path}")
    return path


def run_track_a_and_evaluate(
    models_config: Dict,
    scenarios_data: Dict,
    config: Dict,
    output_dir: Path,
    model_name: str | None = None,
    base_url: str | None = None,
) -> List[Dict]:
    scenarios = scenarios_data.get("scenarios", [])
    conversations = build_conversations_from_scenarios(scenarios)
    if not conversations:
        raise ValueError("No prompts could be generated from scenarios.")

    system_prompts = load_system_prompts(TRACK_ROOT)
    total_turns = sum(len(conv.get("steps", [])) for conv in conversations)
    print(f"Loaded {len(conversations)} conversation scenarios, {total_turns} total turns.")
    print(f"Loaded {len(system_prompts)} system prompts.")

    judge = Judge(config)
    all_results: List[Dict] = []

    for model_cfg in models_config.get("models", []):
        if model_name and model_cfg.get("name") != model_name:
            continue

        provider = model_cfg.get("provider") or model_cfg.get("type")
        if provider not in {"local", "openai-compatible", "remote"}:
            raise ValueError(f"Unsupported provider for {model_cfg.get('name')}: {provider}")

        resolved_base_url = base_url or model_cfg.get("base_url") or model_cfg.get("api_base")
        if not resolved_base_url:
            raise ValueError(f"Model {model_cfg.get('name')} is missing base_url/api_base")

        run_label = safe_name(model_cfg.get("name", "model"))
        model_output_dir = output_dir / run_label
        model_output_dir.mkdir(parents=True, exist_ok=True)
        model_results: List[Dict] = []

        print(f"\n=== Model: {model_cfg.get('name')} ===")
        llm = LocalLLM(
            {
                "name": model_cfg.get("name"),
                "path": model_cfg.get("model_path") or model_cfg.get("path"),
                "quantization": model_cfg.get("quantization", "none"),
                "api_base": resolved_base_url,
                "served_model_name": model_cfg.get("served_model_name") or model_cfg.get("name"),
                "api_key_env": model_cfg.get("api_key_env", "OPENAI_API_KEY"),
                "device": model_cfg.get("device"),
            }
        )

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
                    res = llm.generate([row])[0]
                    res["evaluation"] = judge.evaluate(
                        prompt=res.get("prompt", ""),
                        response=res.get("response", ""),
                        original_prompt=res.get("intent"),
                        scenario=res.get("scenario_description") or res.get("scenario_id"),
                    )
                    model_results.append(res)
                    history = [
                        *history,
                        {"role": "user", "content": step},
                        {"role": "assistant", "content": res.get("response", "")},
                    ]

        del llm
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        save_json(model_results, model_output_dir, f"results_{run_label}")
        all_results.extend(model_results)

    return all_results


def run_divi_clustering(results: List[Dict], output_dir: Path, embedding_model: str) -> None:
    if not results:
        print("No results to cluster; skipping DIVI.")
        return

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import StandardScaler, normalize

    sys.path.append(str(BASE_DIR / "DIVI"))
    from DIVI import DIVIClustering, set_seed  # type: ignore

    texts = [f"{r.get('prompt', '')}\n{r.get('response', '')}" for r in results]
    print(f"Encoding {len(texts)} results with {embedding_model}...")
    embeddings = SentenceTransformer(embedding_model).encode(texts)
    embeddings = np.nan_to_num(embeddings, nan=0.0)
    x_norm = normalize(embeddings, norm="l2")
    x_input = StandardScaler().fit_transform(x_norm)

    d = x_input.shape[1]
    set_seed(42)
    divi = DIVIClustering(split_threshold=None, split_interval=60, max_epochs=200, lr=0.01, verbose=True)
    divi.split_threshold = 0.5 * d * (1 + np.log(2 * np.pi) + np.log(0.9))
    divi.fit(x_input)

    _, _, log_p = divi.model(torch.tensor(x_input, dtype=torch.float32))
    cluster_ids = torch.argmax(log_p, dim=1).detach().cpu().numpy().tolist()

    for row, cid in zip(results, cluster_ids):
        row["cluster"] = int(cid)

    cluster_json = save_json(results, output_dir, "divi_clustered_results")
    df = pd.DataFrame(results)
    csv_path = cluster_json.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved clustering CSV to {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track A + DIVI integration entrypoint")
    parser.add_argument("--models", default=str(BUNDLE_ROOT / "models.yaml"), help="Path to models YAML")
    parser.add_argument(
        "--scenarios",
        default=str(TRACK_ROOT / "config" / "red_team_scenarios copy.json"),
        help="Path to scenarios JSON",
    )
    parser.add_argument("--model_name", default=None, help="Run only one model by name")
    parser.add_argument("--base_url", default=None, help="Override model base_url with OpenAI-compatible endpoint")
    parser.add_argument("--output_dir", default=str(TRACK_ROOT / "data" / "results"), help="Output directory")
    parser.add_argument("--run_divi", action="store_true", help="Run DIVI clustering after generation")
    return parser.parse_args()


def main() -> None:
    load_dotenv(BUNDLE_ROOT / ".env", override=True)
    args = parse_args()

    config = load_yaml(TRACK_ROOT / "config" / "config.yaml")
    models_config = load_yaml(Path(args.models))
    scenarios_data = load_json(Path(args.scenarios))

    results = run_track_a_and_evaluate(
        models_config=models_config,
        scenarios_data=scenarios_data,
        config=config,
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        base_url=args.base_url,
    )

    if not results:
        print("No results generated.")
        return

    aggregate_dir = Path(args.output_dir)
    save_json(results, aggregate_dir, "final_results_aggregated")
    print("Track A run complete.")

    if args.run_divi:
        embedding_model = config.get("embedding_model", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        run_divi_clustering(results, aggregate_dir, embedding_model)


if __name__ == "__main__":
    main()
