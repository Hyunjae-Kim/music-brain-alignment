from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ismir2026.music_task_probe.probing import (  # noqa: E402
    MODEL_REPRESENTATIONS,
    evaluate_embedding_bundle,
    load_embedding_bundle,
    load_final_alignment,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_ROOT = WORKSPACE_ROOT / "noise_music_task_embed" / "embeddings"
DEFAULT_ALIGNMENT_ROOT = WORKSPACE_ROOT / "natural_sound_align"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "noise_music_task_probe" / "probe_results"
DEFAULT_MODELS = ("mulan", "musicgen_melody_large")
DEFAULT_DATASET_DIR = "emo"
DEFAULT_TASK_DIR = "emotion_regression"


def _parse_condition_from_name(condition_id: str) -> tuple[int | None, float | None]:
    if condition_id == "clean":
        return 0, None
    match = re.match(r"^noise_l(?P<level>\d+)_snr_(?P<sign>[pm])(?P<magnitude>[0-9]+p[0-9]+)db$", condition_id)
    if not match:
        return None, None
    level = int(match.group("level"))
    magnitude = float(match.group("magnitude").replace("p", "."))
    snr = magnitude if match.group("sign") == "p" else -magnitude
    return level, snr


def _load_conditions(embeddings_root: Path) -> list[dict[str, Any]]:
    if not embeddings_root.exists():
        raise FileNotFoundError(f"Embeddings root not found: {embeddings_root}")

    manifest_path = embeddings_root / "noise_conditions.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = []
        for row in payload.get("conditions", []):
            rows.append(
                {
                    "noise_condition": str(row["condition_id"]),
                    "noise_level_index": int(row.get("level_index", -1)),
                    "noise_snr_db": row.get("snr_db"),
                }
            )
        return sorted(rows, key=lambda item: (item["noise_level_index"], item["noise_condition"]))

    discovered = []
    for path in sorted(embeddings_root.iterdir()):
        if not path.is_dir():
            continue
        level, snr = _parse_condition_from_name(path.name)
        discovered.append(
            {
                "noise_condition": path.name,
                "noise_level_index": (-1 if level is None else int(level)),
                "noise_snr_db": snr,
            }
        )
    return sorted(discovered, key=lambda item: (item["noise_level_index"], item["noise_condition"]))


def _with_clean_delta(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    clean_rows = (
        summary[summary["noise_condition"] == "clean"]
        [["dataset_name", "task_name", "model_name", "primary_metric_name", "primary_metric_value"]]
        .rename(columns={"primary_metric_value": "clean_primary_metric_value"})
    )
    merged = summary.merge(
        clean_rows,
        on=["dataset_name", "task_name", "model_name", "primary_metric_name"],
        how="left",
    )
    merged["delta_from_clean"] = merged["primary_metric_value"] - merged["clean_primary_metric_value"]
    denom = merged["clean_primary_metric_value"].abs().replace(0.0, np.nan)
    merged["delta_from_clean_pct"] = 100.0 * merged["delta_from_clean"] / denom
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run linear probes for noise-augmented music embeddings.")
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-name",
        action="append",
        choices=tuple(MODEL_REPRESENTATIONS.keys()),
        default=None,
        help="Repeatable. Defaults to mulan + musicgen_melody_large.",
    )
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--task-dir", type=str, default=DEFAULT_TASK_DIR)
    parser.add_argument(
        "--only-condition",
        action="append",
        default=None,
        help="Optional condition_id filter (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    model_names = args.model_name or list(DEFAULT_MODELS)
    conditions = _load_conditions(args.embeddings_root)
    if args.only_condition:
        allowed = set(args.only_condition)
        conditions = [row for row in conditions if row["noise_condition"] in allowed]
        if not conditions:
            raise ValueError(f"No noise conditions matched --only-condition={sorted(allowed)}")

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for condition in conditions:
        condition_root = args.embeddings_root / condition["noise_condition"]
        for model_name in model_names:
            representation_name = MODEL_REPRESENTATIONS[model_name]
            representation_dir = (
                condition_root
                / args.dataset_dir
                / args.task_dir
                / model_name
                / representation_name
            )
            if not representation_dir.exists():
                print(f"skip: missing {representation_dir}", flush=True)
                continue

            bundle = load_embedding_bundle(representation_dir)
            if not bundle:
                print(f"skip: no partitions {representation_dir}", flush=True)
                continue

            try:
                result = evaluate_embedding_bundle(bundle)
            except Exception as exc:  # pragma: no cover - defensive for partial/interrupted extraction outputs
                print(f"skip: invalid bundle {representation_dir} ({exc})", flush=True)
                continue
            summary_row = dict(result["summary"])
            summary_row.update(condition)
            summary_rows.append(summary_row)

            for detail in result["details"]:
                detail_row = dict(detail)
                detail_row.update(condition)
                detail_rows.append(detail_row)

            print(
                f"completed: {condition['noise_condition']} {args.dataset_dir}/{args.task_dir} {model_name}",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    if not summary.empty:
        summary = summary.sort_values(
            ["dataset_name", "task_name", "model_name", "noise_level_index", "noise_condition"],
            kind="stable",
        ).reset_index(drop=True)
    if not details.empty:
        details = details.sort_values(
            ["dataset_name", "task_name", "model_name", "noise_level_index", "noise_condition", "fold_name"],
            kind="stable",
        ).reset_index(drop=True)

    summary = _with_clean_delta(summary)
    alignment = load_final_alignment(args.alignment_root)
    merged = summary.merge(alignment, on="model_name", how="left") if not summary.empty else summary.copy()

    summary.to_csv(args.output_root / "probe_summary.csv", index=False)
    details.to_csv(args.output_root / "probe_fold_details.csv", index=False)
    alignment.to_csv(args.output_root / "alignment_final.csv", index=False)
    merged.to_csv(args.output_root / "probe_summary_with_alignment.csv", index=False)


if __name__ == "__main__":
    main()
