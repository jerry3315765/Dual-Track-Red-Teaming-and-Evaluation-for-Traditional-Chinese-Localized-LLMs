import argparse
import json
from pathlib import Path


def build_dataset(input_path: Path, output_path: Path) -> int:
    with input_path.open('r', encoding='utf-8-sig') as f:
        data = json.load(f)

    scenarios = data.get('scenarios', [])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open('w', encoding='utf-8') as f:
        for scenario in scenarios:
            scenario_id = scenario.get('scenario_id') or scenario.get('id')
            target = scenario.get('target_harmful_output', '')
            steps = [str(s).strip() for s in (scenario.get('steps') or []) if str(s).strip()]
            prompts = [str(p).strip() for p in (scenario.get('prompts') or []) if str(p).strip()]

            # Track B design: single-turn mutated attacks should originate from each
            # step utterance, not from metadata/example fields.
            seeds = steps if steps else prompts
            if not seeds:
                continue
            for idx, prompt in enumerate(seeds):
                scenario_num = scenario_id
                if isinstance(scenario_num, str):
                    digits = "".join(ch for ch in scenario_num if ch.isdigit())
                    scenario_num = int(digits) if digits else 0
                sample_id = int(scenario_num or 0) * 1000 + idx
                row = {
                    'sample_id': sample_id,
                    'scenario_id': scenario_id,
                    'turn_index': idx + 1,
                    'turn_total': len(seeds),
                    'pre_prompt': '',
                    'access_code': target,
                    'post_prompt': '',
                    'attack': prompt,
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description='Build redteam robustness dataset from scenarios JSON')
    parser.add_argument('--input', required=True, help='Path to scenarios JSON')
    parser.add_argument('--output', required=True, help='Path to output JSONL')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    count = build_dataset(input_path, output_path)
    print(f'Wrote {count} rows to {output_path}')


if __name__ == '__main__':
    main()
