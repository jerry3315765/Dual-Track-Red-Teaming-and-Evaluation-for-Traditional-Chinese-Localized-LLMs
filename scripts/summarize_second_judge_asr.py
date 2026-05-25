import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ASR tables from a second-judge CSV")
    parser.add_argument(
        "--input_csv",
        default=str(Path("results") / "second_judge" / "ab_second_judge_full_final.csv"),
        help="Second-judge detailed CSV",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path("results") / "second_judge" / "asr_tables"),
        help="Directory for ASR summary CSVs",
    )
    parser.add_argument(
        "--success_col",
        default="second_result_binary",
        choices=["second_result_binary", "orig_result_binary"],
        help="Binary success column to summarize",
    )
    parser.add_argument(
        "--keep_errors",
        action="store_true",
        help="Keep ERROR rows in denominators. Default drops ERROR/blank second-judge rows.",
    )
    return parser.parse_args()


def as_bool_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).clip(0, 1)


def dedupe_best(df: pd.DataFrame) -> pd.DataFrame:
    if "row_id" not in df.columns:
        return df.copy()
    df = df.copy()
    df["_row_order"] = range(len(df))
    cls = df.get("second_classification", pd.Series("", index=df.index)).fillna("").astype(str)
    result = df.get("second_result_binary", pd.Series("", index=df.index)).fillna("").astype(str)
    df["_valid_second_judge"] = ((cls != "") & (cls != "ERROR") & (result != "")).astype(int)
    df = df.sort_values(["row_id", "_valid_second_judge", "_row_order"])
    df = df.drop_duplicates("row_id", keep="last")
    return df.drop(columns=["_row_order", "_valid_second_judge"], errors="ignore")


def normalize_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "track",
        "model_label",
        "model_name",
        "phase",
        "scenario_id",
        "turn_index",
        "system_prompt_name",
        "attack_method",
        "mutation",
        "query",
        "defense_name",
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["scenario"] = df["scenario_id"]
    df.loc[df["scenario"].eq(""), "scenario"] = df.loc[df["scenario"].eq(""), "query"]
    df.loc[df["scenario"].eq(""), "scenario"] = "unknown"

    df["attack_method_norm"] = df["attack_method"]
    df.loc[df["attack_method_norm"].eq(""), "attack_method_norm"] = df.loc[
        df["attack_method_norm"].eq(""), "mutation"
    ]
    df.loc[df["attack_method_norm"].eq(""), "attack_method_norm"] = df.loc[
        df["attack_method_norm"].eq(""), "system_prompt_name"
    ]
    df.loc[df["attack_method_norm"].eq(""), "attack_method_norm"] = df.loc[
        df["attack_method_norm"].eq(""), "defense_name"
    ]
    df.loc[df["attack_method_norm"].eq(""), "attack_method_norm"] = "unknown"

    df["turn"] = df["turn_index"]
    df.loc[df["turn"].eq(""), "turn"] = "unknown"

    df["phase"] = df["phase"].replace("", "none")
    return df


def summarize(df: pd.DataFrame, group_cols: Iterable[str], success_col: str) -> pd.DataFrame:
    group_cols = list(group_cols)
    grouped = (
        df.groupby(group_cols, dropna=False)[success_col]
        .agg(total="count", success="sum", asr="mean")
        .reset_index()
    )
    grouped["asr_pct"] = (grouped["asr"] * 100).round(2)
    return grouped.sort_values(group_cols).drop(columns=["asr"])


def write_table(df: pd.DataFrame, out_dir: Path, name: str, group_cols: List[str], success_col: str) -> None:
    table = summarize(df, group_cols, success_col)
    table.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv, encoding="utf-8-sig", dtype=str, low_memory=False)
    df = dedupe_best(df_raw)

    cls = df.get("second_classification", pd.Series("", index=df.index)).fillna("").astype(str)
    valid_second = (cls != "") & (cls != "ERROR")
    if not args.keep_errors and args.success_col == "second_result_binary":
        df = df[valid_second].copy()

    df = normalize_dimensions(df)
    df[args.success_col] = as_bool_series(df[args.success_col])

    quality = pd.DataFrame(
        [
            {"metric": "raw_rows", "value": len(df_raw)},
            {"metric": "deduped_rows", "value": len(dedupe_best(df_raw))},
            {"metric": "summarized_rows", "value": len(df)},
            {"metric": "raw_duplicate_extra_rows", "value": len(df_raw) - df_raw.get("row_id", pd.Series()).nunique()},
            {"metric": "raw_error_rows", "value": int((df_raw.get("second_classification", "") == "ERROR").sum())},
        ]
    )
    quality.to_csv(out_dir / "quality_summary.csv", index=False, encoding="utf-8-sig")

    write_table(df, out_dir, "asr_by_track", ["track"], args.success_col)
    write_table(df, out_dir, "asr_by_model", ["model_label"], args.success_col)
    write_table(df, out_dir, "asr_by_track_model", ["track", "model_label"], args.success_col)
    write_table(df, out_dir, "asr_by_track_phase", ["track", "phase"], args.success_col)
    write_table(df, out_dir, "asr_by_scenario", ["track", "scenario"], args.success_col)
    write_table(df, out_dir, "asr_by_turn", ["track", "turn"], args.success_col)
    write_table(df, out_dir, "asr_by_attack_method", ["track", "attack_method_norm"], args.success_col)
    write_table(df, out_dir, "asr_by_track_model_scenario", ["track", "model_label", "scenario"], args.success_col)
    write_table(df, out_dir, "asr_by_track_model_turn", ["track", "model_label", "turn"], args.success_col)
    write_table(
        df,
        out_dir,
        "asr_by_track_model_attack_method",
        ["track", "model_label", "attack_method_norm"],
        args.success_col,
    )

    print(f"[DONE] ASR tables written to: {out_dir}")


if __name__ == "__main__":
    main()
