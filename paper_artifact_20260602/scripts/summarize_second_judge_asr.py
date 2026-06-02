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
        "--summary_csv",
        default="",
        help="Optional model/track summary CSV to write beside the ASR tables.",
    )
    parser.add_argument(
        "--exclude_row_ids_csv",
        default="",
        help="Optional CSV with a row_id column to exclude before summarization.",
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
    parser.add_argument(
        "--keep_empty_responses",
        action="store_true",
        help="Keep rows with blank model responses. Default drops them.",
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


def load_excluded_row_ids(path: str) -> set:
    if not path:
        return set()
    excluded_df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, usecols=["row_id"])
    return set(excluded_df["row_id"].fillna("").astype(str))


def valid_response_mask(series: pd.Series) -> pd.Series:
    response = series.fillna("").astype(str).str.strip()
    return response.ne("") & ~response.str.lower().isin({"nan", "none", "null"})


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


def write_main_summary(df: pd.DataFrame, summary_csv: Path) -> None:
    for col in ["orig_result_binary", "second_result_binary", "agree_binary", "agree_classification"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    summary = (
        df.groupby(["track", "model_label", "model_name", "phase"], dropna=False)
        .agg(
            rows=("row_id", "count"),
            orig_unsafe=("orig_result_binary", "sum"),
            second_unsafe=("second_result_binary", "sum"),
            agree_binary=("agree_binary", "sum"),
            agree_classification=("agree_classification", "sum"),
        )
        .reset_index()
    )
    summary["agree_binary_rate"] = (summary["agree_binary"] / summary["rows"]).round(6)
    summary["agree_classification_rate"] = (summary["agree_classification"] / summary["rows"]).round(6)
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_csv, encoding="utf-8-sig", dtype=str, low_memory=False)
    df = dedupe_best(df_raw)

    excluded_ids = load_excluded_row_ids(args.exclude_row_ids_csv)
    excluded_by_manifest = 0
    if excluded_ids:
        before = len(df)
        df = df[~df["row_id"].fillna("").astype(str).isin(excluded_ids)].copy()
        excluded_by_manifest = before - len(df)

    cls = df.get("second_classification", pd.Series("", index=df.index)).fillna("").astype(str)
    valid_second = (cls != "") & (cls != "ERROR")
    if not args.keep_errors and args.success_col == "second_result_binary":
        df = df[valid_second].copy()

    empty_response_rows = 0
    if not args.keep_empty_responses:
        response_mask = valid_response_mask(df.get("response", pd.Series("", index=df.index)))
        empty_response_rows = int((~response_mask).sum())
        df = df[response_mask].copy()

    df = normalize_dimensions(df)
    df[args.success_col] = as_bool_series(df[args.success_col])

    quality = pd.DataFrame(
        [
            {"metric": "raw_rows", "value": len(df_raw)},
            {"metric": "deduped_rows", "value": len(dedupe_best(df_raw))},
            {"metric": "excluded_by_manifest", "value": excluded_by_manifest},
            {"metric": "invalid_second_judge_rows", "value": int((~valid_second).sum())},
            {"metric": "empty_response_rows", "value": empty_response_rows},
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
    if args.summary_csv:
        write_main_summary(df.copy(), Path(args.summary_csv))

    print(f"[DONE] ASR tables written to: {out_dir}")


if __name__ == "__main__":
    main()
