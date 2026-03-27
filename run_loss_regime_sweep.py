#!/usr/bin/env python3
"""
run_loss_regime_sweep.py

Small controlled sweep for train_comp.py.

For each selected metadata row (one clip/permutation example), this script runs:
    1) audio_l1
    2) audio_mste
    3) cv_l1

Selection:
- stratified over ratio x release_mode
- unique clip_id across the selected set
- same selected examples used for all three regimes

Outputs:
- sweep_runs/<timestamp>/results.csv
- sweep_runs/<timestamp>/summary.csv
- sweep_runs/<timestamp>/summary_latex.txt
- sweep_runs/<timestamp>/selected_examples.csv
- sweep_runs/<timestamp>/ckpts/*.yaml
- sweep_runs/<timestamp>/logs/*.txt
"""

from __future__ import annotations

import argparse
import math
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


@dataclass(frozen=True)
class Regime:
    name: str
    hydra_overrides: List[str]
    paper_label: str


REGIMES: List[Regime] = [
    Regime(
        name="audio_l1",
        paper_label="Waveform L1",
        hydra_overrides=[
            "loss_signal=audio",
            "~loss_fn",
            "+loss_fn={_target_:torch.nn.L1Loss,reduction:mean}",
        ],
    ),
    Regime(
        name="audio_mste",
        paper_label="Waveform MSTE",
        hydra_overrides=[
            "loss_signal=audio",
            "~loss_fn",
            "+loss_fn={_target_:losses.MSTELoss,frame_lengths:[8,16,32,64],overlap:0.75}",
        ],
    ),
    Regime(
        name="cv_l1",
        paper_label="Direct GR L1",
        hydra_overrides=[
            "loss_signal=cv",
            "~loss_fn",
            "+loss_fn={_target_:torch.nn.L1Loss,reduction:mean}",
        ],
    ),
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--metadata", type=Path,
                   default=Path("/Users/bthomp23/Desktop/SSL_CV_dataset/metadata/metadata_main.parquet"))
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/bthomp23/Desktop/SSL_CV_dataset/"),
        help="Root directory against which paths in metadata_main.parquet are resolved.",
    )
    p.add_argument("--trainer", type=Path, default=Path("train_comp.py"))
    p.add_argument("--python", type=str, default=sys.executable)

    p.add_argument(
        "--examples-per-bucket",
        type=int,
        default=2,
        help="Target number of examples per ratio x release_mode bucket.",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Optional hard cap on total examples after stratified selection.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=None)

    p.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable W&B for sweep runs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("sweep_runs"),
    )

    return p.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    if not args.trainer.exists():
        raise FileNotFoundError(f"Trainer file not found: {args.trainer}")


def resolve_data_path(data_root: Path, rel_path: str) -> str:
    return str((data_root / rel_path).resolve())


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    required = [
        "clip_id",
        "permutation_id",
        "audio_in_path",
        "audio_out_path",
        "cv_path",
        "ratio",
        "release_mode",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")

    df = df.copy()
    for c in required:
        df = df[df[c].notna()]

    # Keep one row per exact example if duplicates somehow exist.
    df = df.drop_duplicates(
        subset=["clip_id", "permutation_id", "audio_in_path", "audio_out_path", "cv_path"]
    ).reset_index(drop=True)

    return df


def normalize_release_mode(x) -> str:
    s = str(x).strip().lower()
    if "auto" in s:
        return "auto"
    return "manual"


def make_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ratio_bucket"] = out["ratio"].astype(str)
    out["release_bucket"] = out["release_mode"].map(normalize_release_mode)
    out["bucket"] = list(zip(out["ratio_bucket"], out["release_bucket"]))
    return out


def stratified_unique_clip_sample(
        df: pd.DataFrame,
        examples_per_bucket: int,
        seed: int,
        max_examples: Optional[int] = None,
) -> pd.DataFrame:
    rng = random.Random(seed)
    df = make_bucket_columns(df)

    buckets: List[Tuple[str, str]] = sorted(df["bucket"].drop_duplicates().tolist())
    selected_rows: List[pd.Series] = []
    used_clips: set = set()

    # Shuffle within each bucket for randomness.
    bucket_to_rows: Dict[Tuple[str, str], List[pd.Series]] = {}
    for bucket in buckets:
        sub = df[df["bucket"] == bucket].sample(frac=1.0, random_state=rng.randint(0, 10 ** 9))
        bucket_to_rows[bucket] = [row for _, row in sub.iterrows()]

    # Round-robin selection preserves balance and unique clips.
    rounds = examples_per_bucket
    for _ in range(rounds):
        progress_this_round = False
        for bucket in buckets:
            candidates = bucket_to_rows[bucket]
            chosen_idx = None
            for i, row in enumerate(candidates):
                clip_id = row["clip_id"]
                if clip_id not in used_clips:
                    chosen_idx = i
                    break

            if chosen_idx is None:
                continue

            row = candidates.pop(chosen_idx)
            selected_rows.append(row)
            used_clips.add(row["clip_id"])
            progress_this_round = True

            if max_examples is not None and len(selected_rows) >= max_examples:
                break

        if max_examples is not None and len(selected_rows) >= max_examples:
            break

        if not progress_this_round:
            break

    if not selected_rows:
        raise ValueError("No examples were selected. Check metadata coverage.")

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    # If a hard cap is given and we somehow exceeded it, trim deterministically.
    if max_examples is not None and len(selected) > max_examples:
        selected = (
            selected.sample(frac=1.0, random_state=seed)
            .head(max_examples)
            .sort_values(["ratio_bucket", "release_bucket", "clip_id"])
            .reset_index(drop=True)
        )

    return selected


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def read_ckpt_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint YAML not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected checkpoint format in {path}")
    return data


def build_run_command(
        args: argparse.Namespace,
        row: pd.Series,
        regime: Regime,
        ckpt_path: Path,
) -> List[str]:
    train_input = resolve_data_path(args.data_root, row["audio_in_path"])
    train_target_audio = resolve_data_path(args.data_root, row["audio_out_path"])
    train_target_cv = resolve_data_path(args.data_root, row["cv_path"])

    cmd = [
        args.python,
        str(args.trainer),
        f"data.train.input={train_input}",
        f"data.train.target_audio={train_target_audio}",
        f"data.train.target_cv={train_target_cv}",
        f"data.test.input={train_input}",
        f"data.test.target_audio={train_target_audio}",
        f"data.test.target_cv={train_target_cv}",
        f"ckpt_path={str(ckpt_path.resolve())}",
    ]

    if args.epochs is not None:
        cmd.append(f"epochs={args.epochs}")

    if args.disable_wandb:
        cmd.append("wandb_init.mode=disabled")

    cmd.extend(regime.hydra_overrides)
    return cmd


def run_one(cmd: List[str], log_path: Path, dry_run: bool) -> int:
    print(shell_join(cmd))
    if dry_run:
        return 0

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        print(f"Run failed with return code {proc.returncode}: {log_path}")
    return proc.returncode


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "gr_l1_db",
        "best_loss",
        "esr",
        "test_gr_l1_db",
        "test_loss",
        "test_esr",
    ]

    rows = []
    for regime, g in df.groupby("regime", sort=False):
        row: Dict[str, object] = {
            "regime": regime,
            "paper_label": g["paper_label"].iloc[0],
            "n": len(g),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) else None
            row[f"{col}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0 if len(vals) == 1 else None
        rows.append(row)

    out = pd.DataFrame(rows)

    preferred_order = ["cv_l1", "audio_mste", "audio_l1"]
    if not out.empty:
        out["__order"] = out["regime"].map({k: i for i, k in enumerate(preferred_order)}).fillna(999)
        out = out.sort_values(["__order", "regime"]).drop(columns="__order").reset_index(drop=True)

    return out


def format_mean_std(mean_val, std_val, digits: int = 4) -> str:
    if mean_val is None or pd.isna(mean_val):
        return "--"
    if std_val is None or pd.isna(std_val):
        return f"{mean_val:.{digits}f}"
    return f"{mean_val:.{digits}f} $\\pm$ {std_val:.{digits}f}"


def write_latex_table(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    for _, r in summary_df.iterrows():
        label = r["paper_label"]
        gr = format_mean_std(r.get("gr_l1_db_mean"), r.get("gr_l1_db_std"))
        native = format_mean_std(r.get("best_loss_mean"), r.get("best_loss_std"))
        esr = format_mean_std(r.get("esr_mean"), r.get("esr_std"))
        lines.append(f"{label} & {gr} & {native} & {esr} \\\\")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    validate_paths(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_root / timestamp
    ckpt_dir = run_root / "ckpts"
    log_dir = run_root / "logs"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.metadata)
    selected = stratified_unique_clip_sample(
        df=df,
        examples_per_bucket=args.examples_per_bucket,
        seed=args.seed,
        max_examples=args.max_examples,
    )

    # Nice stable order for inspection
    selected = (
        selected.sort_values(["ratio_bucket", "release_bucket", "clip_id"])
        .reset_index(drop=True)
    )

    selected_csv = run_root / "selected_examples.csv"
    selected.to_csv(selected_csv, index=False)

    print(f"Selected {len(selected)} examples")
    print(selected[["clip_id", "permutation_id", "ratio", "release_mode"]].to_string(index=False))
    print(f"Saved selection to: {selected_csv}")

    results: List[Dict[str, object]] = []

    for _, row in selected.iterrows():
        clip_id = str(row["clip_id"])
        permutation_id = str(row["permutation_id"])

        for regime in REGIMES:
            run_id = f"{clip_id}_{permutation_id}_{regime.name}"
            ckpt_path = ckpt_dir / f"{run_id}.yaml"
            log_path = log_dir / f"{run_id}.txt"

            cmd = build_run_command(
                args=args,
                row=row,
                regime=regime,
                ckpt_path=ckpt_path,
            )

            return_code = run_one(cmd=cmd, log_path=log_path, dry_run=args.dry_run)

            result_row: Dict[str, object] = {
                "run_id": run_id,
                "clip_id": clip_id,
                "permutation_id": permutation_id,
                "ratio": row["ratio"],
                "release_mode": row["release_mode"],
                "paper_label": regime.paper_label,
                "regime": regime.name,
                "audio_in_path": row["audio_in_path"],
                "audio_out_path": row["audio_out_path"],
                "cv_path": row["cv_path"],
                "ckpt_path": str(ckpt_path),
                "log_path": str(log_path),
                "return_code": return_code,
                "best_loss": None,
                "gr_l1_db": None,
                "esr": None,
                "test_loss": None,
                "test_gr_l1_db": None,
                "test_esr": None,
                "threshold": None,
                "ratio_fit": None,
                "attack_ms": None,
                "release_ms": None,
                "make_up_gain": None,
            }

            if return_code == 0 and not args.dry_run:
                try:
                    ckpt = read_ckpt_yaml(ckpt_path)
                    formatted = ckpt.get("formated_params", {}) or {}

                    result_row.update(
                        {
                            "best_loss": safe_float(ckpt.get("loss")),
                            "gr_l1_db": safe_float(ckpt.get("gr_l1_db")),
                            "esr": safe_float(ckpt.get("esr")),
                            "test_loss": safe_float(ckpt.get("test_loss")),
                            "test_gr_l1_db": safe_float(ckpt.get("test_gr_l1_db")),
                            "test_esr": safe_float(ckpt.get("test_esr")),
                            "threshold": safe_float(ckpt.get("threshold")),
                            "ratio_fit": safe_float(formatted.get("ratio")),
                            "attack_ms": safe_float(formatted.get("attack_ms")),
                            "release_ms": safe_float(formatted.get("release_ms")),
                            "make_up_gain": safe_float(ckpt.get("make_up_gain")),
                        }
                    )
                except Exception as e:
                    result_row["parse_error"] = str(e)

            results.append(result_row)
            pd.DataFrame(results).to_csv(run_root / "results.csv", index=False)

    results_df = pd.DataFrame(results)
    results_csv = run_root / "results.csv"
    summary_csv = run_root / "summary.csv"
    latex_txt = run_root / "summary_latex.txt"

    results_df.to_csv(results_csv, index=False)

    ok = results_df[results_df["return_code"] == 0].copy()
    summary_df = aggregate_results(ok)
    summary_df.to_csv(summary_csv, index=False)
    write_latex_table(summary_df, latex_txt)

    print("\nDone.")
    print(f"Run root:      {run_root}")
    print(f"Results CSV:   {results_csv}")
    print(f"Summary CSV:   {summary_csv}")
    print(f"LaTeX rows:    {latex_txt}")

    if not summary_df.empty:
        print("\nSummary:")
        print(
            summary_df[
                [
                    "paper_label",
                    "n",
                    "gr_l1_db_mean",
                    "gr_l1_db_std",
                    "best_loss_mean",
                    "best_loss_std",
                    "esr_mean",
                    "esr_std",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
