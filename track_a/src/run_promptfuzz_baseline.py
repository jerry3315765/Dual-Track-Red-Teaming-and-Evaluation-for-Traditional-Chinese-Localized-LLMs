import os
import yaml
import subprocess
import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TRACK_ROOT = BASE_DIR.parent
BUNDLE_ROOT = TRACK_ROOT.parent

def run_promptfuzz_on_models(models_yaml, promptfuzz_dir):
    with open(models_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    models = config.get('models', [])
    
    # We will test the first model by default or let user choose.
    print(f"Found {len(models)} models in {models_yaml}")
    
    for model in models:
        model_name = model['name']
        model_url = model.get('base_url') or model.get('api_base')
        if not model_url:
            raise ValueError(f"Model {model_name} is missing base_url/api_base")
        if not model_url.endswith('/v1'):
            model_url = model_url.rstrip('/') + '/v1'

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or not api_key.startswith('sk-'):
            raise ValueError('OPENAI_API_KEY must be a real OpenAI API key')
            
        print(f"--- Running PromptFuzz Baseline for {model_name} ---")
        
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = model_url
        env["OPENAI_API_KEY"] = api_key
        
        # Command to run PromptFuzz's main script
        # mode=redteam, phase=focus
        run_script = os.path.abspath(os.path.join(promptfuzz_dir, "Experiment", "run.py"))
        promptfuzz_dir_abs = os.path.abspath(promptfuzz_dir)
        
        cmd = [
            "python", run_script,
            "--mode", "redteam",
            "--phase", "focus",
            "--model_path", model['path'],  # the model to query
            "--openai_key", api_key,
            "--max_query", "50", # Reduced for fast baseline testing
            "--energy", "1"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        # We use subprocess
        try:
            # We don't block fully in this example script if there are issues, just show one.
            subprocess.run(cmd, env=env, cwd=promptfuzz_dir_abs, check=True)
            # print("[Info] Uncomment subprocess.run in src/run_promptfuzz_baseline.py to actually execute.")
            # print("[Info] Make sure `gptfuzzer` is installed in your python environment before running.")
        except subprocess.CalledProcessError as e:
            print(f"Error running PromptFuzz: {e}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=str(BUNDLE_ROOT / "models.yaml"))
    parser.add_argument("--promptfuzz_dir", type=str, default="PromptFuzz-Thesis")
    args = parser.parse_args()
    
    run_promptfuzz_on_models(args.models, args.promptfuzz_dir)
