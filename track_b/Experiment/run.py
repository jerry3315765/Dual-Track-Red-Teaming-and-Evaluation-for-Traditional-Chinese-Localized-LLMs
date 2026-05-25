import sys
import os
import json
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add the track_b root to sys.path
TRACK_B_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUNDLE_ROOT = os.path.abspath(os.path.join(TRACK_B_ROOT, '..'))
sys.path.append(TRACK_B_ROOT)

from PromptFuzz.Fuzzer.promptfuzz import run_fuzzer
from PromptFuzz.utils import constants

def load_models_config(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return yaml.safe_load(f)

def apply_model_config(args, models_config):
    models = models_config.get('models', [])
    selected = None
    if args.model_name:
        for m in models:
            if m.get('name') == args.model_name:
                selected = m
                break
    elif args.model_index is not None and 0 <= args.model_index < len(models):
        selected = models[args.model_index]

    if not selected:
        raise ValueError("Model not found in models.yaml. Use --model_name or --model_index.")

    args.model_path = selected.get('served_model_name') or selected.get('model_path') or selected.get('path')
    args.base_url = args.base_url or selected.get('base_url')
    args.baseline = selected.get('name') if args.baseline is None else args.baseline

    api_key_env = selected.get('api_key_env')
    if api_key_env:
        args.openai_key = os.getenv(api_key_env) or args.openai_key

    args.selected_model_name = selected.get('name')
    args.selected_model_quantization = selected.get('quantization', 'unknown')
    args.selected_model_path = selected.get('model_path') or selected.get('path')

def build_redteam_dataset_if_needed(scenarios_path, output_path):
    from Experiment.build_redteam_robustness_dataset import build_dataset
    return build_dataset(Path(scenarios_path), Path(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fuzzing parameters')
    parser.add_argument('--index', type=int, default=0, help='The index of the target prompt')
    parser.add_argument('--phase', choices=['preparation', 'focus', 'init'], default='init', help='The phase of the fuzzing process')
    parser.add_argument('--mode', choices=['hijacking', 'extraction', 'redteam'], default='redteam', help='The mode of the fuzzing process')
    parser.add_argument('--openai_key', type=str, default=None, help='OpenAI API Key')
    parser.add_argument('--base_url', type=str, default=None, help='Base URL for OpenAI API')
    parser.add_argument('--model_path', type=str, default='gpt-3.5-turbo-0125', help='target model path')
    parser.add_argument('--models_config', type=str, default=os.path.abspath(os.path.join(TRACK_B_ROOT, '..', 'models.yaml')),
                        help='Path to models.yaml')
    parser.add_argument('--model_name', type=str, default=None, help='Model name in models.yaml')
    parser.add_argument('--model_index', type=int, default=None, help='Model index in models.yaml')
    parser.add_argument('--scenarios', type=str, default=os.path.abspath(os.path.join(TRACK_B_ROOT, '..', 'track_a', 'config', 'red_team_scenarios copy.json')),
                        help='Path to scenarios.json')
    parser.add_argument('--results_root', type=str, default=os.path.join(BUNDLE_ROOT, 'results', 'track_b'),
                        help='Root directory for Track B results')
    parser.add_argument('--run_label', type=str, default=None, help='Optional label for output folders')
    parser.add_argument('--max_query', type=int, default=None,
                        help='The maximum number of queries')
    parser.add_argument('--max_jailbreak', type=int,
                        default=9999999, help='The maximum jailbreak number')
    parser.add_argument('--target_max_tokens', type=int, default=4096,
                        help='Max output tokens for target model generation in Track B')
    parser.add_argument('--mutator_max_tokens', type=int, default=512,
                        help='Max output tokens for mutator model generation in Track B')
    parser.add_argument('--energy', type=int, default=1,
                        help='The energy of the fuzzing process')
    parser.add_argument("--no_mutate", action='store_true', help='Do not mutate the prompt')
    parser.add_argument("--all_defenses", action='store_true', help='Run all defenses')
    parser.add_argument("--concatenate", action='store_true', help='Concatenate the prompt')
    parser.add_argument("--few_shot", action='store_true', help='Use few shot learning')
    parser.add_argument("--retrieval_method", choices=['random', 'cosine_similarity', 'cluster'], default='random', help='The retrieval method')
    parser.add_argument("--cluster_num", type=int, default=5, help='The number of clusters for retrieval')
    parser.add_argument("--few_shot_num", type=int, default=3)
    parser.add_argument("--dynamic_allocate", action='store_true', help='Dynamic allocate the energy')
    parser.add_argument("--threshold_coefficient", type=float, default=0.5, help='The threshold coefficient')
    parser.add_argument("--mutator_weights", type=float, nargs='+', default=[0.2, 0.2, 0.2, 0.2, 0.2], help='The weights for the mutator selection')
    parser.add_argument("--baseline", type=str, default=None, help='The baseline model')

    args = parser.parse_args()
    args.max_query_explicit = args.max_query is not None

    # Keep runtime-injected env (shell/parent process) higher priority than .env defaults.
    load_dotenv(os.path.join(BUNDLE_ROOT, '.env'), override=False)
    
    if args.openai_key is None:
        args.openai_key = constants.openai_key

    if args.models_config and os.path.exists(args.models_config):
        models_config = load_models_config(args.models_config)
        apply_model_config(args, models_config)

    if args.scenarios and os.path.exists(args.scenarios) and args.mode == 'redteam' and args.phase == 'init':
        output_path = os.path.join(TRACK_B_ROOT, 'Datasets', 'redteam_robustness_dataset.jsonl')
        build_redteam_dataset_if_needed(args.scenarios, output_path)

    os.makedirs(args.results_root, exist_ok=True)
    if args.run_label is None:
        args.run_label = args.selected_model_name
        
    run_fuzzer(args)
