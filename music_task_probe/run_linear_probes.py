from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ismir2026.music_task_probe.probing import (
    build_probe_jobs,
    evaluate_embedding_bundle,
    load_embedding_bundle,
    load_final_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple linear probes over prepared music task embeddings.")
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path("/home/hjkim/selective_auditory/ismir2026/music_task_embed/embeddings"),
        help="Root directory that stores extracted embeddings.",
    )
    parser.add_argument(
        "--alignment-root",
        type=Path,
        default=Path("/home/hjkim/selective_auditory/ismir2026/natural_sound_align"),
        help="Root directory for RSA alignment summaries.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/hjkim/selective_auditory/ismir2026/music_task_probe/probe_results"),
        help="Directory for probe CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for job in build_probe_jobs(args.embeddings_root):
        representation_dir = Path(job["representation_dir"])
        bundle = load_embedding_bundle(representation_dir)
        result = evaluate_embedding_bundle(bundle)
        summary_rows.append(result["summary"])
        detail_rows.extend(result["details"])
        print(f"completed: {job['dataset_dir']}/{job['task_dir']} {job['model_name']}", flush=True)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset_name", "task_name", "model_name"],
        kind="stable",
    )
    details = pd.DataFrame(detail_rows).sort_values(
        ["dataset_name", "task_name", "model_name", "fold_name"],
        kind="stable",
    )
    alignment = load_final_alignment(args.alignment_root)
    merged = summary.merge(alignment, on="model_name", how="left")

    summary.to_csv(args.output_root / "probe_summary.csv", index=False)
    details.to_csv(args.output_root / "probe_fold_details.csv", index=False)
    alignment.to_csv(args.output_root / "alignment_final.csv", index=False)
    merged.to_csv(args.output_root / "probe_summary_with_alignment.csv", index=False)


if __name__ == "__main__":
    main()
