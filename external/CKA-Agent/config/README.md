## Main Configuration (`config.yml`)

This file is the primary entry point for defining an experiment. It is divided into several key sections:

-   **`experiment`**: General settings like the experiment name and output directory.
-   **`data`**: Specifies the dataset(s) to be used for the experiment.
-   **`model`**: Defines the target model to be attacked.
-   **`methods`**: Lists the jailbreak methods to be run.
-   **`evaluation`**: Configures the evaluation process, including judge models and metrics.
-   **`defense`**: Specifies any defense mechanisms to be applied to the target model.

### Detailed Configuration (`config.yml`)

#### Experiment Settings
```yaml
experiment:
  name: "my_jailbreak_experiment" # A unique name for your experiment
  output_dir: "results/"         # Directory to save experiment outputs
```

#### Data Configuration
```yaml
data:
  dataset_names: ["harmbench_cka"] # Available datasets: ["harmbench_cka", "strongreject_cka"]
  # You can specify multiple datasets, e.g., ["harmbench_cka", "strongreject_cka"]
  # ... other data settings like batch_size, num_workers etc.
```

#### Model Configuration
```yaml
model:
  target_model: "gemini-2.5-xxx" # Example black-box models: "gpt-oss-120b", "claude-xxx"
  # Example white-box models (requires local setup): "llama-2-7b", "vicuna-13b"
  # You can also specify model paths for local models
  # ... other model-specific parameters like temperature, top_p
```

#### Methods Configuration
```yaml
methods:
  baselines:
    - "cka-agent"       # CKA-Agent method for correlated knowledge jailbreak
    # - "actor_attack"   # Actor Attack (multi-turn role-playing jailbreak)
    # xxx
    # choose to uncomment one of the above methods to enable it
```

#### Evaluation Configuration
```yaml
evaluation:
  judge_model: "gemini-2.5-flash" # Model used to evaluate jailbreak success (e.g., "llama-70b")
  metrics: ["attack_success_rate", "harmful_rate"] # Metrics to calculate
  # ... other evaluation settings like batch_size
```

#### Defense Configuration
```yaml
defense:
  enabled: false          # Set to true to apply a defense mechanism
  type: "llm_guard"       # Available defenses: "llm_guard", "perturbation_defense", "rephrasing_defense", "grayswanai_guard"
  # ... other defense-specific parameters
```

## Method-Specific Configurations (`method/*.yml`)

Each jailbreak method listed in the `methods` section of `config.yml` has its own configuration file in the `config/method/` directory. This allows for detailed, method-specific parameter tuning without cluttering the main configuration file.

For example, if you enable the `cka-agent` method in `config.yml`, its behavior is controlled by `config/method/cka-agent.yml`. This file might contain settings for internal models, hyperparameters for the attack algorithm, and other specific parameters.

By editing these files, you can precisely control each aspect of your jailbreak experiment.
