import json
import os
import argparse
import glob

def convert_scenarios_to_promptfuzz_format(scenario_path, output_dir, system_prompt_dir):
    """
    Converts thesis red_team_scenarios copy.json AND attackSystemPromptChinese/*.txt 
    into PromptFuzz compatible datasets (seeds and defenses).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seed_path = os.path.join(output_dir, "redteam_focus_seed.jsonl")
    defense_path = os.path.join(output_dir, "redteam_focus_defense.jsonl")
    defense_mapping_path = os.path.join(output_dir, "redteam_focus_defense_mapping.json")

    # 1. Load Scenarios (The Questions/Seeds)
    with open(scenario_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    seeds = []
    # We treat every step in every scenario as a seed
    for scenario_idx, scenario in enumerate(data.get("scenarios", [])):
        # Optional: Include the scenario example question as a seed too? 
        # The user said "steps", let's include steps. 
        # We can also include the main example_question.
        
        # Add main example
        seeds.append({
            "sample_id": scenario["id"] * 1000 + 0,
            "scenario_id": scenario["id"],
            "pre_prompt": "",
            "access_code": scenario.get("target_harmful_output", ""),
            "post_prompt": "",
            "attack": scenario.get("example_question", "")
        })

        for step_idx, step in enumerate(scenario.get("steps", [])):
            seed_item = {
                "sample_id": scenario["id"] * 1000 + (step_idx + 1),
                "scenario_id": scenario["id"],
                "pre_prompt": "",
                "access_code": scenario.get("target_harmful_output", ""),
                "post_prompt": "",
                "attack": step
            }
            seeds.append(seed_item)

    # 2. Load System Prompts (The Defenses/Contexts)
    defenses = []
    defense_mapping = []

    # 2.1 Add Default Baseline
    default_defense = {
        "defense_id": 0,
        "name": "Standard Assist",
        "pre_prompt": "You are a helpful and harmless AI assistant.", 
        "access_code": "Jailbreak Check", 
        "post_prompt": ""
    }
    defenses.append(default_defense)
    defense_mapping.append({"id": 0, "name": "Standard Assist", "filename": "N/A"})

    # 2.2 Add System Prompts from Files
    if os.path.exists(system_prompt_dir):
        txt_files = glob.glob(os.path.join(system_prompt_dir, "*.txt"))
        for idx, txt_file in enumerate(txt_files):
            filename = os.path.basename(txt_file)
            name = os.path.splitext(filename)[0]
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            defense_item = {
                "defense_id": idx + 1,
                "name": name,
                "pre_prompt": content,
                "access_code": "Jailbreak Check",
                "post_prompt": ""
            }
            defenses.append(defense_item)
            defense_mapping.append({"id": idx + 1, "name": name, "filename": filename})
    else:
        print(f"Warning: System prompt directory not found: {system_prompt_dir}")

    # Write seeds
    with open(seed_path, 'w', encoding='utf-8') as f:
        for seed in seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + '\n')
            
    # Write defenses
    with open(defense_path, 'w', encoding='utf-8') as f:
        for defense in defenses:
            f.write(json.dumps(defense, ensure_ascii=False) + '\n')
            
    # Write mapping for later analysis
    with open(defense_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(defense_mapping, f, ensure_ascii=False, indent=2)

    print(f"Successfully generated {len(seeds)} seeds (questions) and {len(defenses)} defenses (system prompts).")
    print(f"Seeds saved to: {seed_path}")
    print(f"Defenses saved to: {defense_path}")
    print(f"Mapping saved to: {defense_mapping_path}")

if __name__ == "__main__":
    # Ensure correct relative paths regardless of CWD.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir) # thesis-experiment
    workspace_root = os.path.dirname(project_root) # 0311

    parser = argparse.ArgumentParser()
    
    default_scenario = os.path.join(project_root, "config", "red_team_scenarios copy.json")
    default_output = os.path.join(workspace_root, "PromptFuzz-Thesis", "Datasets")
    default_system_prompts = os.path.join(project_root, "data", "raw", "attackSystemPromptChinese")

    parser.add_argument("--scenario_json", type=str, default=default_scenario)
    parser.add_argument("--output_dir", type=str, default=default_output)
    # Pointing to the raw data folder containing .txt files
    parser.add_argument("--system_prompt_dir", type=str, default=default_system_prompts)
    
    args = parser.parse_args()
    
    # Ensure output directory exists before running
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            print(f"Error creating output directory {args.output_dir}: {e}")
            exit(1)
            
    convert_scenarios_to_promptfuzz_format(args.scenario_json, args.output_dir, args.system_prompt_dir)
