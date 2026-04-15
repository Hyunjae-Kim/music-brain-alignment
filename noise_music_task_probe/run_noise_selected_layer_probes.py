from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ismir2026.music_task_probe.probing import evaluate_embedding_bundle, load_embedding_bundle  # noqa: E402


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOISE_ROOT = WORKSPACE_ROOT / "noise_music_task_embed" / "embeddings"
DEFAULT_CLEAN_ROOT = WORKSPACE_ROOT / "music_task_embed" / "embeddings"
DEFAULT_ALIGNMENT_LAYERWISE = WORKSPACE_ROOT / "music_task_probe" / "probe_results_all_models" / "alignment_layerwise.csv"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "noise_music_task_probe" / "probe_results_selected_layers"
DEFAULT_DATASET_DIR = "emo"
DEFAULT_TASK_DIR = "emotion_regression"
DEFAULT_MODEL_LAYER = (
    "mulan:layer_08",
    "musicgen_melody_large:layer_35",
)
DEFAULT_CONDITIONS = (
    "noise_l1_snr_p20p0db",
    "noise_l2_snr_p10p0db",
    "noise_l3_snr_p0p0db",
)


def _parse_model_layer(entries: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Expected model:layer format, got {entry}")
        model, layer = entry.split(":", 1)
        parsed.append((model.strip().lower(), layer.strip()))
    return parsed


def _condition_meta(condition: str) -> tuple[int, float | None]:
    if condition == "clean":
        return 0, None
    m = re.match(r"^noise_l(\d+)_snr_([pm])(\d+)p(\d+)db$", condition)
    if m is None:
        return -1, None
    level = int(m.group(1))
    val = float(f"{m.group(3)}.{m.group(4)}")
    if m.group(2) == "m":
        val = -val
    return level, val


def _clean_representation_dir(
    clean_root: Path,
    dataset_dir: str,
    task_dir: str,
    model_name: str,
    layer_name: str,
) -> Path:
    return clean_root / dataset_dir / task_dir / model_name / layer_name


def _noise_representation_dir(
    noise_root: Path,
    condition: str,
    dataset_dir: str,
    task_dir: str,
    model_name: str,
    layer_name: str,
) -> Path:
    return noise_root / condition / dataset_dir / task_dir / model_name / layer_name


def _attach_clean_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    clean = (
        summary[summary["noise_condition"] == "clean"]
        [["model_name", "representation_name", "primary_metric_name", "primary_metric_value"]]
        .rename(columns={"primary_metric_value": "clean_primary_metric_value"})
    )
    merged = summary.merge(
        clean,
        on=["model_name", "representation_name", "primary_metric_name"],
        how="left",
    )
    merged["delta_from_clean"] = merged["primary_metric_value"] - merged["clean_primary_metric_value"]
    denom = merged["clean_primary_metric_value"].abs().replace(0.0, np.nan)
    merged["delta_from_clean_pct"] = 100.0 * merged["delta_from_clean"] / denom
    return merged


def _load_alignment_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        [
            "alignment_kind",
            "alignment_dataset",
            "model_name",
            "alignment_representation_name",
            "alignment_score",
        ]
    ].copy()
    frame = frame.rename(columns={"alignment_representation_name": "representation_name"})
    frame["model_name"] = frame["model_name"].astype(str).str.lower()

    pivot = (
        frame.pivot_table(
            index=["model_name", "representation_name", "alignment_dataset"],
            columns="alignment_kind",
            values="alignment_score",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"rsa": "alignment_rsa", "regression": "alignment_regression"})
    )

    avg = (
        pivot[pivot["alignment_dataset"] == "mean"]
        [["model_name", "representation_name", "alignment_rsa", "alignment_regression"]]
        .rename(
            columns={
                "alignment_rsa": "alignment_rsa_mean",
                "alignment_regression": "alignment_regression_mean",
            }
        )
    )

    by_ds = pivot[pivot["alignment_dataset"].isin(["NH2015", "B2021"])].copy()
    by_ds = by_ds.pivot_table(
        index=["model_name", "representation_name"],
        columns="alignment_dataset",
        values=["alignment_rsa", "alignment_regression"],
        aggfunc="mean",
    )
    by_ds.columns = [f"{metric}_{dataset}" for metric, dataset in by_ds.columns]
    by_ds = by_ds.reset_index()

    return avg.merge(by_ds, on=["model_name", "representation_name"], how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe selected best layers under clean + staged noise.")
    parser.add_argument("--noise-embeddings-root", type=Path, default=DEFAULT_NOISE_ROOT)
    parser.add_argument("--clean-embeddings-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--alignment-layerwise", type=Path, default=DEFAULT_ALIGNMENT_LAYERWISE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--task-dir", type=str, default=DEFAULT_TASK_DIR)
    parser.add_argument("--model-layer", action="append", default=list(DEFAULT_MODEL_LAYER))
    parser.add_argument("--condition", action="append", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    model_layers = _parse_model_layer(args.model_layer)
    conditions = list(dict.fromkeys(args.condition or []))
    if args.include_clean and "clean" not in conditions:
        conditions = ["clean", *conditions]

    rows: list[dict[str, Any]] = []

    for model_name, layer_name in model_layers:
        for condition in conditions:
            if condition == "clean":
                rep_dir = _clean_representation_dir(
                    args.clean_embeddings_root,
                    args.dataset_dir,
                    args.task_dir,
                    model_name,
                    layer_name,
                )
            else:
                rep_dir = _noise_representation_dir(
                    args.noise_embeddings_root,
                    condition,
                    args.dataset_dir,
                    args.task_dir,
                    model_name,
                    layer_name,
                )

            if not rep_dir.exists():
                print(f"skip: missing {rep_dir}", flush=True)
                continue

            bundle = load_embedding_bundle(rep_dir)
            if not bundle:
                print(f"skip: empty {rep_dir}", flush=True)
                continue

            try:
                result = evaluate_embedding_bundle(bundle)
            except Exception as exc:
                print(f"skip: invalid {rep_dir} ({exc})", flush=True)
                continue

            level, snr_db = _condition_meta(condition)
            row = dict(result["summary"])
            row.update(
                {
                    "noise_condition": condition,
                    "noise_level_index": int(level),
                    "noise_snr_db": snr_db,
                }
            )
            rows.append(row)
            print(f"completed: {condition} {model_name} {layer_name}", flush=True)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["model_name", "representation_name", "noise_level_index", "noise_condition"],
            kind="stable",
        ).reset_index(drop=True)
    summary = _attach_clean_deltas(summary)

    alignment = _load_alignment_map(args.alignment_layerwise)
    merged = summary.merge(alignment, on=["model_name", "representation_name"], how="left")

    summary.to_csv(args.output_root / "selected_layer_probe_summary.csv", index=False)
    merged.to_csv(args.output_root / "selected_layer_probe_summary_with_alignment.csv", index=False)


if __name__ == "__main__":
    main()
