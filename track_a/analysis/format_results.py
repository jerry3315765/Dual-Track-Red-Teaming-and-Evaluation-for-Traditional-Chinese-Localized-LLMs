import os
import json
import yaml
import glob
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
EXPORT_DIR = os.path.join(BASE_DIR, 'data', 'exported_by_model')

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_export():
    print("Starting export of model results...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # 1. Find all result files (grouped by prompt)
    result_files = glob.glob(os.path.join(RESULTS_DIR, "redteam_divi_results_*.json"))
    
    if not result_files:
        print("No result files found in data/results/")
        return

    # 2. Iterate and regroup
    # Structure: data_by_model[model_name][prompt_name] = [list of results]
    data_by_model = defaultdict(lambda: defaultdict(list))
    
    total_records = 0
    
    for file_path in result_files:
        filename = os.path.basename(file_path)
        # Extract prompt name from filename: redteam_divi_results_{prompt_name}.json
        prompt_name = filename.replace("redteam_divi_results_", "").replace(".json", "")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for record in data:
                model_name = record.get('model', 'unknown_model')
                data_by_model[model_name][prompt_name].append(record)
                total_records += 1
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print(f"Processed {total_records} records across {len(result_files)} files.")

    # 3. Write to new files
    for model_name, prompts_data in data_by_model.items():
        model_dir = os.path.join(EXPORT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"\nExporting for Model: {model_name}")
        
        for prompt_name, records in prompts_data.items():
            output_filename = f"{model_name}_{prompt_name}.json"
            output_path = os.path.join(model_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            
            print(f"  - Saved {len(records)} records to {output_filename}")

    print(f"\nExport complete! Files are in: {EXPORT_DIR}")

if __name__ == "__main__":
    run_export()
