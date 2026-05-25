import subprocess
from pathlib import Path
import yaml


def run(cmd, cwd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    bundle_root = Path(__file__).resolve().parents[1]
    models_path = bundle_root / "models.yaml"
    runner = bundle_root / "scripts" / "run_single_model.py"

    with models_path.open("r", encoding="utf-8") as f:
        models_config = yaml.safe_load(f)

    for model in models_config.get("models", []):
        name = model.get("name")
        if not name:
            continue
        run(
            [
                "python",
                str(runner),
                "--model_name",
                name,
            ],
            cwd=bundle_root,
        )


if __name__ == "__main__":
    main()
