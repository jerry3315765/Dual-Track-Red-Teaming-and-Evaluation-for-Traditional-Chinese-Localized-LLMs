# Automatic Red Team Testing and Evaluation for Traditional Chinese Localized LLMs

This repository contains the code, data, and analysis scripts for the thesis experiment: **Automatic Red Team Testing and Evaluation for Traditional Chinese Localized LLMs**.

The project implements a complete automated red-teaming pipeline, including multi-turn adversarial dialogue, LLM-as-a-judge evaluation (using GPT-4o-mini), and a post-hoc mechanistic diagnosis using the DIVI algorithm combined with SHAP values.

## Project Structure

```text
thesis-experiment/
├── analysis/               # Scripts for statistical analysis, ASR calculation, and SHAP analysis
├── config/                 # YAML and JSON configurations for models, prompts, and scenarios
├── data/                   # Experimental data
│   ├── raw/                # System prompts, adversarial templates, and raw intents
│   └── processed/          # Processed statistics, human audit samples, and merged results
├── paper/                  # LaTeX source code of the thesis/paper
├── results/                # Output directory for generated traces, Divi clusters, and metrics
│   ├── divi_clusters/      # Results from DIVI clustering and SHAP analysis
│   ├── exported/           # Model responses and generation outputs
│   └── raw_traces/         # Saved multi-turn conversation traces
├── src/                    # Core pipeline implementation
│   ├── core/               # Evaluation and judge implementation
│   ├── DIVI/               # Gumbel-Sigmoid ELBO implementation for clustering
│   ├── integrate_redteam_divi.py  # Main script binding red-teaming and DIVI
│   ├── main.py             # Red-teaming test runner
│   └── retry_bad_samples.py # Error-handling and retry logic for LM inference
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Quick Start

1. **Setup Environment**:  
   We recommend using a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   pip install -r requirements.txt
   ```

2. **Configure API Keys & Models**:  
   Create a `.env` file and place your OpenAI API key for the Judge:
   ```env
   OPENAI_API_KEY=your_key_here
   ```
   Modify `config/models.yaml` to point to your local LM Studio instances or HuggingFace endpoints.

3. **Run Red Teaming Experiments**:  
   ```bash
   python src/integrate_redteam_divi.py
   # or
   python src/main.py
   ```

4. **Analyze Results**:  
   Run scripts in the `analysis/` folder to generate SHAP reports, success rates (ASR), and visual clusters:
   ```bash
   python analysis/analyze_success_rates.py
   python analysis/analyze_shap_clusters.py
   ```

## License

This project is licensed under the MIT License.