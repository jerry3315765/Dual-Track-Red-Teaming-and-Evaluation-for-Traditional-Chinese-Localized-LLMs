import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import paramiko
import requests
import yaml


BUNDLE_ROOT = Path(__file__).resolve().parent
TRACK_A_ROOT = BUNDLE_ROOT / "track_a"
TRACK_B_ROOT = BUNDLE_ROOT / "track_b"
DEFAULT_MODELS = BUNDLE_ROOT / "models.yaml"
DEFAULT_SCENARIOS = TRACK_A_ROOT / "config" / "red_team_scenarios copy.json"
DEFAULT_RESULTS_ROOT = BUNDLE_ROOT / "results"

RUNPOD_HOST = "64.247.201.55"
RUNPOD_PORT = 12635
RUNPOD_USER = "root"
SSH_KEY_PATH = r"C:\Users\jerry\.ssh\id_ed25519"
RUNPOD_BASE_URL = "https://yiiifq98p9jc95-8888.proxy.runpod.net/v1"
VLLM_PORT = 8888


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models(path: Path) -> List[Dict]:
    return list(load_yaml(path).get("models", []))


def safe_name(value: str) -> str:
    keep = []
    for ch in str(value):
        keep.append(ch if (ch.isalnum() or ch in "._-") else "_")
    return "".join(keep)


def load_private_key(key_path: str, passphrase: str | None):
    loaders = [paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey]
    last_err = None
    for loader in loaders:
        try:
            return loader.from_private_key_file(key_path, password=passphrase)
        except Exception as err:
            last_err = err
    raise last_err


def connect_ssh(host: str, port: int, user: str, key_path: str, passphrase: str | None) -> paramiko.SSHClient:
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"SSH key not found: {key_path}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pkey = load_private_key(key_path, passphrase)
    client.connect(hostname=host, port=port, username=user, pkey=pkey, timeout=30)
    transport = client.get_transport()
    if transport is not None:
        # Keep the SSH session alive during long local compute phases.
        transport.set_keepalive(30)
    return client


def run_ssh_command(client: paramiko.SSHClient, command: str) -> Tuple[str, str, int]:
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    status = stdout.channel.recv_exit_status()
    if err:
        print(f"[SSH stderr] {err.strip()}")
    return out, err, status


def stop_vllm(client: paramiko.SSHClient, port: int) -> None:
    print(f"Cleaning up remote vLLM and port {port}...")
    run_ssh_command(client, "pkill -f vllm")
    run_ssh_command(client, f"lsof -ti:{port} | xargs -r kill -9")
    time.sleep(3)


def stop_vllm_best_effort(
    client: paramiko.SSHClient,
    port: int,
    *,
    ssh_host: str,
    ssh_port: int,
    ssh_user: str,
    ssh_key: str,
    ssh_passphrase: str | None,
) -> paramiko.SSHClient:
    try:
        stop_vllm(client, port)
        return client
    except Exception as e:
        print(f"[WARN] stop_vllm failed on existing SSH session: {e}")
        try:
            client.close()
        except Exception:
            pass
        try:
            fresh = connect_ssh(ssh_host, ssh_port, ssh_user, ssh_key, ssh_passphrase)
            stop_vllm(fresh, port)
            return fresh
        except Exception as e2:
            print(f"[WARN] stop_vllm retry also failed, skipping cleanup: {e2}")
            return client


def build_vllm_command(model_cfg: Dict, port: int) -> str:
    model_path = model_cfg.get("model_path") or model_cfg.get("path")
    if not model_path:
        raise ValueError(f"Missing model_path/path for {model_cfg.get('name')}")
    served_model_name = model_cfg.get("served_model_name") or model_cfg.get("name")
    dtype = model_cfg.get("vllm_dtype") or model_cfg.get("dtype") or "bfloat16"
    max_model_len = int(model_cfg.get("max_model_len") or 8000)
    extra_args = model_cfg.get("vllm_args") or []
    if isinstance(extra_args, str):
        extra_args = [extra_args]
    if not any("trust-remote-code" in str(arg) for arg in extra_args):
        extra_args.append("--trust-remote-code")

    parts = [
        "nohup",
        "env",
        "VLLM_USE_DEEP_GEMM=0",
        "python3 -m vllm.entrypoints.openai.api_server",
        f"--model {shlex.quote(str(model_path))}",
        f"--served-model-name {shlex.quote(str(served_model_name))}",
        f"--dtype {shlex.quote(str(dtype))}",
        f"--max-model-len {max_model_len}",
        f"--port {port}",
    ]
    parts.extend(extra_args)
    parts.append("> /root/vllm.log 2>&1 &")
    return " ".join(parts)


def maybe_patch_breeze2_config(client: paramiko.SSHClient, model_cfg: Dict) -> None:
    model_name = str(model_cfg.get("name") or "").lower()
    model_path = str(model_cfg.get("model_path") or model_cfg.get("path") or "")
    if "breeze2-8b" not in model_name or not model_path:
        return

    target_file = f"{model_path.rstrip('/')}/configuration_internvl_chat.py"
    print(f"[Patch] Applying compatibility patch for breeze2 config: {target_file}")

    patch_script = f"""
from pathlib import Path

p = Path({json.dumps(target_file)})
if not p.exists():
    raise FileNotFoundError(f"Patch target not found: {{p}}")

text = p.read_text(encoding="utf-8")
if "llm_architectures = llm_config.get('architectures')" in text:
    print("Patch already applied")
    raise SystemExit(0)

old = \"\"\"        self.vision_config = InternVisionConfig(**vision_config)
        if llm_config['architectures'][0] == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        elif llm_config['architectures'][0] == 'MistralForCausalLM':
            self.llm_config = MistralConfig(**llm_config)
        else:
            raise ValueError('Unsupported architecture: {{}}'.format(llm_config['architectures'][0]))
\"\"\"

new = \"\"\"        self.vision_config = InternVisionConfig(**vision_config)
        llm_architectures = llm_config.get('architectures') or ['LlamaForCausalLM']
        llm_arch = llm_architectures[0]
        if llm_arch == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_arch == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        elif llm_arch == 'MistralForCausalLM':
            self.llm_config = MistralConfig(**llm_config)
        else:
            logger.warning('Unsupported architecture %s; fallback to LlamaConfig', llm_arch)
            self.llm_config = LlamaConfig(**llm_config)
\"\"\"

if old not in text:
    raise RuntimeError("Expected snippet not found; patch pattern mismatch")

bak = p.with_suffix(p.suffix + ".bak")
if not bak.exists():
    bak.write_text(text, encoding="utf-8")

p.write_text(text.replace(old, new), encoding="utf-8")
print("Patch applied")
"""
    remote_cmd = f"python3 - <<'PY'\n{patch_script}\nPY"
    _, _, status = run_ssh_command(client, f"bash -lc {shlex.quote(remote_cmd)}")
    if status != 0:
        raise RuntimeError("Failed to patch breeze2 configuration_internvl_chat.py")


def start_vllm(client: paramiko.SSHClient, model_cfg: Dict, port: int) -> None:
    stop_vllm(client, port)
    maybe_patch_breeze2_config(client, model_cfg)
    print(f"Starting model: {model_cfg.get('name')}...")
    _, _, status = run_ssh_command(client, "python3 -c 'import vllm'")
    if status != 0:
        raise RuntimeError("Remote python3 cannot import vLLM. Install vLLM on the pod first.")
    command = build_vllm_command(model_cfg, port)
    run_ssh_command(client, f"bash -lc {shlex.quote(command)}")


def wait_for_vllm(base_url: str, expected_model: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    url = f"{base_url}/models"
    last_ids = []
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                last_ids = [x.get("id", "") for x in data]
                if expected_model in last_ids:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM ready timeout. expected={expected_model}, last_ids={last_ids}")


def run_python(command: List[str], cwd: Path, extra_env: Dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def track_a_paths(job: Dict, output_root: Path) -> Tuple[Path, Path]:
    model_out = output_root / "track_a" / job["label"]
    raw_results_path = model_out / f"raw_results_{job['label']}.json"
    evaluated_results_path = model_out / f"results_{job['label']}.json"
    return raw_results_path, evaluated_results_path


def track_b_phase_result_dir(job: Dict, output_root: Path, phase: str) -> Path:
    # Track B writes by:
    # <results_root>/<run_label>/<phase>/redteam/baseline/<model_name>/all_results_long.csv
    model_name = str(job["model"].get("name") or "")
    return (
        output_root
        / "track_b"
        / job["label"]
        / phase
        / "redteam"
        / "baseline"
        / model_name
    )


def is_track_b_phase_done(job: Dict, output_root: Path, phase: str) -> bool:
    result_dir = track_b_phase_result_dir(job, output_root, phase)
    return is_nonempty_file(result_dir / "all_results.csv") and is_nonempty_file(result_dir / "all_results_long.csv")


def run_track_a(job: Dict, args: argparse.Namespace, scenarios: Path, output_root: Path) -> Path:
    out = output_root / "track_a"
    model_out = out / job["label"]
    model_out.mkdir(parents=True, exist_ok=True)
    raw_results_path = model_out / f"raw_results_{job['label']}.json"
    cmd = [
        args.python_exe,
        str(TRACK_A_ROOT / "src" / "main.py"),
        "--mode",
        "generate",
        "--models",
        str(job["config_path"]),
        "--scenarios",
        str(scenarios),
        "--model_name",
        job["model"]["name"],
        "--run_label",
        job["label"],
        "--output_dir",
        str(out),
        "--base_url",
        args.base_url,
        "--raw_results_path",
        str(raw_results_path),
    ]
    run_python(cmd, cwd=TRACK_A_ROOT, extra_env={"OPENAI_API_KEY": args.judge_api_key})
    return raw_results_path


def run_track_a_evaluate(
    job: Dict,
    args: argparse.Namespace,
    scenarios: Path,
    output_root: Path,
    input_results_path: Path,
) -> None:
    out = output_root / "track_a"
    model_out = out / job["label"]
    model_out.mkdir(parents=True, exist_ok=True)
    evaluated_results_path = model_out / f"results_{job['label']}.json"
    cmd = [
        args.python_exe,
        str(TRACK_A_ROOT / "src" / "main.py"),
        "--mode",
        "evaluate",
        "--models",
        str(job["config_path"]),
        "--scenarios",
        str(scenarios),
        "--model_name",
        job["model"]["name"],
        "--run_label",
        job["label"],
        "--output_dir",
        str(out),
        "--base_url",
        args.base_url,
        "--input_results_path",
        str(input_results_path),
        "--evaluated_results_path",
        str(evaluated_results_path),
    ]
    run_python(cmd, cwd=TRACK_A_ROOT, extra_env={"OPENAI_API_KEY": args.judge_api_key})


def run_track_b(job: Dict, args: argparse.Namespace, scenarios: Path, output_root: Path, phases: Iterable[str]) -> None:
    out = output_root / "track_b"
    out.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        args.python_exe,
        str(TRACK_B_ROOT / "Experiment" / "run.py"),
        "--mode",
        "redteam",
        "--all_defenses",
        "--models_config",
        str(job["config_path"]),
        "--model_name",
        job["model"]["name"],
        "--run_label",
        job["label"],
        "--scenarios",
        str(scenarios),
        "--base_url",
        args.base_url,
        "--results_root",
        str(out),
        "--target_max_tokens",
        str(args.track_b_target_max_tokens),
        "--openai_key",
        args.judge_api_key,
    ]
    if args.track_b_max_query is not None:
        base_cmd.extend(["--max_query", str(args.track_b_max_query)])

    for phase in phases:
        cmd = [*base_cmd, "--phase", phase]
        if phase == "init":
            cmd.append("--no_mutate")
        run_python(cmd, cwd=TRACK_B_ROOT, extra_env={"OPENAI_API_KEY": args.judge_api_key})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Track A + Track B full experiment")
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS))
    p.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--base_url", default=RUNPOD_BASE_URL)
    p.add_argument("--vllm_port", type=int, default=VLLM_PORT)
    p.add_argument("--vllm_timeout", type=int, default=600)
    p.add_argument("--model_name", default=None)
    p.add_argument("--model_index", type=int, default=None)
    p.add_argument("--model_start_index", type=int, default=None, help="Run models starting from this index (inclusive)")
    p.add_argument("--model_end_index", type=int, default=None, help="Run models up to this index (inclusive)")
    p.add_argument("--skip_track_a", action="store_true")
    p.add_argument("--skip_track_b", action="store_true")
    p.add_argument("--track_b_phase", choices=["both", "init", "focus"], default="both")
    p.add_argument("--track_b_max_query", type=int, default=None)
    p.add_argument("--track_b_target_max_tokens", type=int, default=4096)
    p.add_argument("--track_b_smoke", action="store_true")
    p.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume mode. By default completed outputs are detected and skipped.",
    )
    p.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to next model when one model fails.",
    )
    p.add_argument(
        "--disable_parallel_track_a_eval",
        action="store_true",
        help="Disable overlap of Track A local evaluation with Track B fuzzing",
    )
    p.add_argument("--reuse_vllm", action="store_true")
    p.add_argument("--ssh_host", default=RUNPOD_HOST)
    p.add_argument("--ssh_port", type=int, default=RUNPOD_PORT)
    p.add_argument("--ssh_user", default=RUNPOD_USER)
    p.add_argument("--ssh_key", default=SSH_KEY_PATH)
    p.add_argument("--ssh_passphrase_env", default="RUNPOD_SSH_PASSPHRASE")
    p.add_argument("--judge_api_key", default=os.getenv("OPENAI_API_KEY", ""))
    p.add_argument("--python_exe", default=os.getenv("PYTHON_EXE", str(Path(sys.executable))))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = Path(args.scenarios)
    if not scenarios.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenarios}")

    models = load_models(Path(args.models))
    jobs = []
    for m in models:
        if args.model_name and m.get("name") != args.model_name:
            continue
        jobs.append(
            {
                "model": m,
                "config_path": Path(args.models),
                "label": f"{safe_name(m.get('name', 'model'))}__{safe_name(m.get('precision') or m.get('quantization') or 'default')}",
            }
        )
    if args.model_index is not None:
        jobs = [jobs[args.model_index]]
    else:
        if args.model_start_index is not None:
            if args.model_start_index < 0 or args.model_start_index >= len(jobs):
                raise ValueError(f"--model_start_index {args.model_start_index} out of range for {len(jobs)} models")
            jobs = jobs[args.model_start_index:]
        if args.model_end_index is not None:
            if args.model_end_index < 0:
                raise ValueError("--model_end_index must be >= 0")
            jobs = jobs[: args.model_end_index + 1]
    if not jobs:
        raise ValueError("No model jobs selected.")

    phases = ["init", "focus"] if args.track_b_phase == "both" else [args.track_b_phase]
    if args.track_b_smoke:
        args.skip_track_a = True
        phases = ["init"]
        if args.track_b_max_query is None:
            args.track_b_max_query = 1

    output_root = Path(args.results_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resume_enabled = not args.no_resume

    passphrase = os.getenv(args.ssh_passphrase_env)
    client = connect_ssh(args.ssh_host, args.ssh_port, args.ssh_user, args.ssh_key, passphrase)
    failed_models: List[Tuple[str, str]] = []
    try:
        for job in jobs:
            model = job["model"]
            model_name = str(model.get("name"))
            expected = model.get("served_model_name") or model.get("name")
            print(f"\n=== Running model: {model_name} ===")

            try:
                raw_results_path, evaluated_results_path = track_a_paths(job, output_root)
                track_a_generate_done = is_nonempty_file(raw_results_path)
                track_a_evaluate_done = is_nonempty_file(evaluated_results_path)
                phase_done_map = {
                    phase: is_track_b_phase_done(job, output_root, phase)
                    for phase in phases
                }
                pending_track_b_phases = [phase for phase in phases if not phase_done_map[phase]]

                if resume_enabled:
                    if track_a_generate_done:
                        print(f"[Resume] Track A generate already exists: {raw_results_path}")
                    if track_a_evaluate_done:
                        print(f"[Resume] Track A evaluate already exists: {evaluated_results_path}")
                    for phase in phases:
                        if phase_done_map[phase]:
                            print(f"[Resume] Track B phase '{phase}' already exists")

                need_track_a = not args.skip_track_a and (
                    (not resume_enabled) or (not track_a_evaluate_done)
                )
                need_track_b = not args.skip_track_b and (
                    (not resume_enabled) or bool(pending_track_b_phases)
                )

                if not need_track_a and not need_track_b:
                    print("[Resume] Model already complete. Skipping.")
                    continue

                if not args.reuse_vllm:
                    start_vllm(client, model, args.vllm_port)
                wait_for_vllm(args.base_url, expected, args.vllm_timeout)

                if need_track_a and need_track_b and not args.disable_parallel_track_a_eval:
                    if not (resume_enabled and track_a_generate_done):
                        raw_results_path = run_track_a(job, args, scenarios, output_root)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        track_b_future = executor.submit(
                            run_track_b,
                            job,
                            args,
                            scenarios,
                            output_root,
                            pending_track_b_phases if resume_enabled else phases,
                        )
                        if not (resume_enabled and track_a_evaluate_done):
                            run_track_a_evaluate(job, args, scenarios, output_root, raw_results_path)
                        track_b_future.result()
                else:
                    if need_track_a and (args.disable_parallel_track_a_eval or not need_track_b):
                        if not (resume_enabled and track_a_generate_done):
                            raw_results_path = run_track_a(job, args, scenarios, output_root)
                        if not (resume_enabled and track_a_evaluate_done):
                            run_track_a_evaluate(job, args, scenarios, output_root, raw_results_path)
                    if need_track_b:
                        run_track_b(
                            job,
                            args,
                            scenarios,
                            output_root,
                            pending_track_b_phases if resume_enabled else phases,
                        )

                if not args.reuse_vllm:
                    client = stop_vllm_best_effort(
                        client,
                        args.vllm_port,
                        ssh_host=args.ssh_host,
                        ssh_port=args.ssh_port,
                        ssh_user=args.ssh_user,
                        ssh_key=args.ssh_key,
                        ssh_passphrase=passphrase,
                    )
            except Exception as exc:
                failed_models.append((model_name, str(exc)))
                print(f"[ERROR] Model failed: {model_name}: {exc}")
                if not args.reuse_vllm:
                    client = stop_vllm_best_effort(
                        client,
                        args.vllm_port,
                        ssh_host=args.ssh_host,
                        ssh_port=args.ssh_port,
                        ssh_user=args.ssh_user,
                        ssh_key=args.ssh_key,
                        ssh_passphrase=passphrase,
                    )
                if not args.continue_on_error:
                    raise
                print("[Continue] continue_on_error enabled, moving to next model.")
    finally:
        try:
            client.close()
        except Exception:
            pass

    if failed_models:
        print("\n=== Failed Models Summary ===")
        for name, msg in failed_models:
            print(f"- {name}: {msg}")


if __name__ == "__main__":
    main()
