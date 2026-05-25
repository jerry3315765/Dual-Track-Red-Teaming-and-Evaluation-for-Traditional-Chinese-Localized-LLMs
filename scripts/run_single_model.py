import argparse
import subprocess
from pathlib import Path


def run(cmd, cwd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run Track A + Track B for a single model")
    parser.add_argument("--model_name", required=True, help="Model name in models.yaml")
    parser.add_argument("--skip_track_a", action="store_true")
    parser.add_argument("--skip_track_b", action="store_true")
    parser.add_argument("--skip_focus", action="store_true")
    parser.add_argument("--skip_init", action="store_true")
    args = parser.parse_args()

    bundle_root = Path(__file__).resolve().parents[1]
    models_path = bundle_root / "models.yaml"
    scenarios_path = bundle_root / "track_a" / "config" / "red_team_scenarios copy.json"

    if not args.skip_track_a:
        run(
            [
                "python",
                str(bundle_root / "track_a" / "src" / "main.py"),
                "--models",
                str(models_path),
                "--scenarios",
                str(scenarios_path),
                "--model_name",
                args.model_name,
            ],
            cwd=bundle_root,
        )

    if not args.skip_track_b:
        if not args.skip_init:
            run(
                [
                    "python",
                    str(bundle_root / "track_b" / "Experiment" / "run.py"),
                    "--phase",
                    "init",
                    "--mode",
                    "redteam",
                    "--all_defenses",
                    "--no_mutate",
                    "--models_config",
                    str(models_path),
                    "--scenarios",
                    str(scenarios_path),
                    "--model_name",
                    args.model_name,
                ],
                cwd=bundle_root / "track_b",
            )

        if not args.skip_focus:
            run(
                [
                    "python",
                    str(bundle_root / "track_b" / "Experiment" / "run.py"),
                    "--phase",
                    "focus",
                    "--mode",
                    "redteam",
                    "--all_defenses",
                    "--models_config",
                    str(models_path),
                    "--model_name",
                    args.model_name,
                ],
                cwd=bundle_root / "track_b",
            )


if __name__ == "__main__":
    main()
