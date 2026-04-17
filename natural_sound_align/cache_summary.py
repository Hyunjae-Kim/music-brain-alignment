from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import PACKAGE_ROOT, load_tuckute_stimulus_metadata, normalize_dataset_name
from .embeddings import available_layer_indices, final_representation_layer_depth
from .pipeline import DEFAULT_MODELS, DEFAULT_OUTPUT_ROOT, run_all_models, run_single_model_analysis


COMPILED_ROOT = PACKAGE_ROOT / "compiled_results"
FOUNDATION_MODELS = (
    "CLAP",
    "CLAP_MUSIC",
    "CLAP_MUSIC_SPEECH",
    "MULAN",
    "MERT",
    "MERT95",
    "WAV2VEC2",
    "JUKEBOX",
    "MUSICGEN_SMALL",
    "MUSICGEN_MEDIUM",
    "MUSICGEN_LARGE",
    "MUSICGEN_MELODY_LARGE",
)


@dataclass(frozen=True)
class MetricSummaryBundle:
    base_analysis: dict[str, object]
    regression_summary: pd.DataFrame
    rsa_summary: pd.DataFrame
    layerwise_tables: dict[str, pd.DataFrame]


def iter_representation_specs(model_name: str) -> list[tuple[int | None, str]]:
    model_name = model_name.strip().upper()
    if model_name == "SPECTEMP":
        return [(None, "avgpool")]
    specs = [(None, "final")]
    specs.extend((layer_index, f"layer_{layer_index:02d}") for layer_index in available_layer_indices(model_name))
    return specs


def ensure_full_cache(
    *,
    datasets: Iterable[str] = ("NH2015", "B2021"),
    models: Iterable[str] = DEFAULT_MODELS,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> None:
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    for dataset_name in datasets:
        normalized_dataset = normalize_dataset_name(dataset_name)
        for model_name in models:
            for layer_index, _ in iter_representation_specs(model_name):
                run_single_model_analysis(
                    model_name,
                    dataset_name=normalized_dataset,
                    layer_index=layer_index,
                    output_root=output_root,
                    overwrite=overwrite,
                )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _final_layer_depth(model_name: str) -> int:
    return final_representation_layer_depth(model_name)


def _parse_representation_metadata(model_name: str, representation_name: str) -> tuple[int | None, int]:
    if representation_name == "final":
        return None, _final_layer_depth(model_name)
    if representation_name == "avgpool":
        return None, 0
    if representation_name.startswith("layer_"):
        layer_index = int(representation_name.split("_", 1)[1])
        return layer_index, layer_index
    return None, _final_layer_depth(model_name)


def collect_compiled_summaries(
    *,
    output_root: Path | None = None,
    datasets: Iterable[str] = ("NH2015", "B2021"),
    allowed_models: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    normalized_allowed_models = {model_name.strip().upper() for model_name in allowed_models} if allowed_models else None
    regression_rows: list[dict[str, object]] = []
    rsa_rows: list[dict[str, object]] = []

    for dataset_name in datasets:
        normalized_dataset = normalize_dataset_name(dataset_name)
        dataset_root = output_root / normalized_dataset.lower()
        if not dataset_root.exists():
            continue

        for model_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            if normalized_allowed_models is not None and model_dir.name.upper() not in normalized_allowed_models:
                continue
            for representation_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                regression_summary_path = representation_dir / "regression_summary.json"
                rsa_summary_path = representation_dir / "rsa_summary.json"
                rsa_metrics_path = representation_dir / "rsa_metrics.npz"
                if not regression_summary_path.exists() or not rsa_summary_path.exists() or not rsa_metrics_path.exists():
                    continue

                representation_name = representation_dir.name
                layer_index, layer_depth = _parse_representation_metadata(model_dir.name, representation_name)
                is_final = representation_name == "final"

                regression_row = _load_json(regression_summary_path)
                regression_row.update(
                    {
                        "dataset_name": normalized_dataset,
                        "layer_index": layer_index,
                        "layer_depth": layer_depth,
                        "is_final": is_final,
                        "result_dir": str(representation_dir),
                    }
                )
                regression_rows.append(regression_row)

                rsa_row = _load_json(rsa_summary_path)
                rsa_row.update(
                    {
                        "dataset_name": normalized_dataset,
                        "layer_index": layer_index,
                        "layer_depth": layer_depth,
                        "is_final": is_final,
                        "result_dir": str(representation_dir),
                    }
                )
                rsa_rows.append(rsa_row)

    regression_df = pd.DataFrame(regression_rows).sort_values(
        ["dataset_name", "model_name", "layer_depth", "representation_name"],
        ignore_index=True,
    )
    rsa_df = pd.DataFrame(rsa_rows).sort_values(
        ["dataset_name", "model_name", "layer_depth", "representation_name"],
        ignore_index=True,
    )
    return regression_df, rsa_df


def write_compiled_summaries(
    *,
    output_root: Path | None = None,
    compiled_root: Path | None = None,
    datasets: Iterable[str] = ("NH2015", "B2021"),
    allowed_models: Iterable[str] | None = None,
    merge_existing: bool = False,
) -> tuple[Path, Path]:
    compiled_root = compiled_root or COMPILED_ROOT
    compiled_root.mkdir(parents=True, exist_ok=True)
    if allowed_models is None:
        allowed_models = (*FOUNDATION_MODELS, "SPECTEMP")
    regression_df, rsa_df = collect_compiled_summaries(
        output_root=output_root,
        datasets=datasets,
        allowed_models=allowed_models,
    )

    regression_csv = compiled_root / "regression_layerwise_summary.csv"
    rsa_csv = compiled_root / "rsa_layerwise_summary.csv"
    regression_pkl = compiled_root / "regression_layerwise_summary.pkl"
    rsa_pkl = compiled_root / "rsa_layerwise_summary.pkl"

    if merge_existing and regression_csv.exists() and rsa_csv.exists():
        existing_regression = pd.read_csv(regression_csv)
        existing_rsa = pd.read_csv(rsa_csv)

        def _normalized_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
            key_frame = frame.copy()
            key_frame["_dataset_key"] = key_frame["dataset_name"].astype(str).str.upper()
            key_frame["_model_key"] = key_frame["model_name"].astype(str).str.upper()
            key_frame["_rep_key"] = key_frame["representation_name"].astype(str)
            return key_frame

        incoming_regression = _normalized_key_frame(regression_df)
        incoming_rsa = _normalized_key_frame(rsa_df)
        existing_regression_keys = _normalized_key_frame(existing_regression)
        existing_rsa_keys = _normalized_key_frame(existing_rsa)

        incoming_regression_key_set = set(
            zip(
                incoming_regression["_dataset_key"],
                incoming_regression["_model_key"],
                incoming_regression["_rep_key"],
            )
        )
        incoming_rsa_key_set = set(
            zip(
                incoming_rsa["_dataset_key"],
                incoming_rsa["_model_key"],
                incoming_rsa["_rep_key"],
            )
        )

        keep_regression_mask = [
            (dataset_key, model_key, rep_key) not in incoming_regression_key_set
            for dataset_key, model_key, rep_key in zip(
                existing_regression_keys["_dataset_key"],
                existing_regression_keys["_model_key"],
                existing_regression_keys["_rep_key"],
            )
        ]
        keep_rsa_mask = [
            (dataset_key, model_key, rep_key) not in incoming_rsa_key_set
            for dataset_key, model_key, rep_key in zip(
                existing_rsa_keys["_dataset_key"],
                existing_rsa_keys["_model_key"],
                existing_rsa_keys["_rep_key"],
            )
        ]

        regression_df = pd.concat([existing_regression.loc[keep_regression_mask], regression_df], ignore_index=True)
        rsa_df = pd.concat([existing_rsa.loc[keep_rsa_mask], rsa_df], ignore_index=True)

    regression_df = regression_df.sort_values(
        ["dataset_name", "model_name", "layer_depth", "representation_name"],
        ignore_index=True,
        kind="stable",
    )
    rsa_df = rsa_df.sort_values(
        ["dataset_name", "model_name", "layer_depth", "representation_name"],
        ignore_index=True,
        kind="stable",
    )

    regression_df.to_csv(regression_csv, index=False)
    rsa_df.to_csv(rsa_csv, index=False)
    regression_df.to_pickle(regression_pkl)
    rsa_df.to_pickle(rsa_pkl)
    return regression_pkl, rsa_pkl


def load_compiled_summaries(
    *,
    compiled_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    compiled_root = compiled_root or COMPILED_ROOT
    regression_pkl = compiled_root / "regression_layerwise_summary.pkl"
    rsa_pkl = compiled_root / "rsa_layerwise_summary.pkl"
    if not regression_pkl.exists() or not rsa_pkl.exists():
        raise FileNotFoundError(
            "Compiled summaries are missing. Run ensure_full_cache() and write_compiled_summaries() first."
        )
    return pd.read_pickle(regression_pkl), pd.read_pickle(rsa_pkl)


def select_plot_summary(
    summary_df: pd.DataFrame,
    *,
    dataset_name: str,
    layer_mode: str,
    metric_column: str,
) -> pd.DataFrame:
    normalized_dataset = normalize_dataset_name(dataset_name)
    subset = summary_df.loc[summary_df["dataset_name"] == normalized_dataset].copy()
    if layer_mode not in {"final", "best"}:
        raise ValueError("layer_mode must be either 'final' or 'best'.")

    baseline = subset.loc[subset["model_name"] == "SPECTEMP"].copy()
    if layer_mode == "final":
        foundation = subset.loc[
            (subset["model_name"] != "SPECTEMP") & subset["is_final"].astype(bool)
        ].copy()
    else:
        foundation = (
            subset.loc[subset["model_name"] != "SPECTEMP"]
            .sort_values(metric_column, ascending=False)
            .groupby("model_name", as_index=False)
            .head(1)
            .copy()
        )

    merged = pd.concat([foundation, baseline], ignore_index=True)
    return merged.sort_values(metric_column, ascending=False, ignore_index=True)


def load_cached_rsa_payload(
    *,
    dataset_name: str,
    model_name: str,
    representation_name: str,
    output_root: Path | None = None,
) -> dict[str, object]:
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    normalized_dataset = normalize_dataset_name(dataset_name)
    result_dir = output_root / normalized_dataset.lower() / model_name.lower() / representation_name
    rsa_summary = _load_json(result_dir / "rsa_summary.json")
    with np.load(result_dir / "rsa_metrics.npz", allow_pickle=True) as npz_file:
        payload = {
            "summary": rsa_summary,
            "model_correlation_matrix": np.asarray(npz_file["model_correlation_matrix"], dtype=np.float64),
            "human_correlation_matrices": np.asarray(npz_file["human_correlation_matrices"], dtype=np.float64),
            "mean_human_correlation_matrix": np.asarray(npz_file["mean_human_correlation_matrix"], dtype=np.float64),
            "participant_ids": np.asarray(npz_file["participant_ids"], dtype=int),
            "split_scores": np.asarray(npz_file["split_scores"], dtype=np.float64),
        }
    payload["metadata"] = load_tuckute_stimulus_metadata(normalized_dataset)
    payload["result_dir"] = result_dir
    return payload


def _artifact_to_layer_row(
    artifact: dict[str, object],
    *,
    layer_index: int | None,
    available_layers: list[int],
) -> dict[str, object]:
    regression_summary = dict(artifact["regression_summary"])
    rsa_summary = dict(artifact["rsa_summary"])
    regression_summary.pop("dataset_name", None)
    rsa_summary.pop("dataset_name", None)

    representation_name = str(regression_summary["representation_name"])
    return {
        "model_name": str(regression_summary["model_name"]),
        "representation_name": representation_name,
        "layer_index": None if layer_index is None else int(layer_index),
        "layer_depth": max(available_layers) + 1 if layer_index is None and available_layers else (0 if layer_index is None else int(layer_index)),
        "is_final": layer_index is None,
        "cv_subject_mean": float(regression_summary["cv_subject_mean"]),
        "cv_subject_median": float(regression_summary["cv_subject_median"]),
        "median_noise_corrected_r2": float(regression_summary["median_noise_corrected_r2"]),
        "mean_noise_corrected_r2": float(regression_summary["mean_noise_corrected_r2"]),
        "mean_spearman": float(rsa_summary["mean_spearman"]),
        "median_spearman": float(rsa_summary["median_spearman"]),
        "n_participants": int(rsa_summary["n_participants"]),
    }


def collect_layerwise_metrics(
    model_name: str,
    *,
    dataset_name: str,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    available_layers = available_layer_indices(model_name)
    rows: list[dict[str, object]] = []

    final_artifact = run_single_model_analysis(
        model_name,
        dataset_name=dataset_name,
        output_root=output_root,
        overwrite=overwrite,
    )
    rows.append(
        _artifact_to_layer_row(
            final_artifact,
            layer_index=None,
            available_layers=available_layers,
        )
    )

    for layer_index in available_layers:
        artifact = run_single_model_analysis(
            model_name,
            dataset_name=dataset_name,
            layer_index=layer_index,
            output_root=output_root,
            overwrite=overwrite,
        )
        rows.append(
            _artifact_to_layer_row(
                artifact,
                layer_index=layer_index,
                available_layers=available_layers,
            )
        )

    return pd.DataFrame(rows).sort_values("layer_depth").reset_index(drop=True)


def _best_row(layerwise_table: pd.DataFrame, metric_column: str) -> pd.Series:
    return layerwise_table.sort_values(metric_column, ascending=False).iloc[0]


def build_metric_summaries(
    *,
    dataset_name: str,
    layer_mode: str = "final",
    output_root: Path | None = None,
    overwrite: bool = False,
    regression_metric: str = "cv_subject_mean",
    rsa_metric: str = "median_spearman",
) -> MetricSummaryBundle:
    if layer_mode not in {"final", "best"}:
        raise ValueError("layer_mode must be either 'final' or 'best'.")

    output_root = output_root or DEFAULT_OUTPUT_ROOT
    base_analysis = run_all_models(
        dataset_name=dataset_name,
        output_root=output_root,
        overwrite=overwrite,
    )
    regression_summary = base_analysis["regression_summary"].copy().reset_index(drop=True)
    rsa_summary = base_analysis["rsa_summary"].copy().reset_index(drop=True)
    layerwise_tables: dict[str, pd.DataFrame] = {}

    if layer_mode == "final":
        return MetricSummaryBundle(
            base_analysis=base_analysis,
            regression_summary=regression_summary,
            rsa_summary=rsa_summary,
            layerwise_tables=layerwise_tables,
        )

    regression_rows = [regression_summary.query("model_name == 'SPECTEMP'").iloc[0].to_dict()]
    rsa_rows = [rsa_summary.query("model_name == 'SPECTEMP'").iloc[0].to_dict()]

    for model_name in FOUNDATION_MODELS:
        layerwise_table = collect_layerwise_metrics(
            model_name,
            dataset_name=dataset_name,
            output_root=output_root,
            overwrite=overwrite,
        )
        layerwise_tables[model_name] = layerwise_table

        best_regression = _best_row(layerwise_table, regression_metric)
        regression_rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "representation_name": best_regression["representation_name"],
                "cv_subject_mean": float(best_regression["cv_subject_mean"]),
                "cv_subject_median": float(best_regression["cv_subject_median"]),
                "median_noise_corrected_r2": float(best_regression["median_noise_corrected_r2"]),
                "mean_noise_corrected_r2": float(best_regression["mean_noise_corrected_r2"]),
            }
        )

        best_rsa = _best_row(layerwise_table, rsa_metric)
        rsa_rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "representation_name": best_rsa["representation_name"],
                "mean_spearman": float(best_rsa["mean_spearman"]),
                "median_spearman": float(best_rsa["median_spearman"]),
                "n_participants": int(best_rsa["n_participants"]),
            }
        )

    regression_plot_summary = pd.DataFrame(regression_rows).sort_values(
        regression_metric,
        ascending=False,
    ).reset_index(drop=True)
    rsa_plot_summary = pd.DataFrame(rsa_rows).sort_values(
        rsa_metric,
        ascending=False,
    ).reset_index(drop=True)

    return MetricSummaryBundle(
        base_analysis=base_analysis,
        regression_summary=regression_plot_summary,
        rsa_summary=rsa_plot_summary,
        layerwise_tables=layerwise_tables,
    )


def select_model_artifact(
    model_name: str,
    *,
    dataset_name: str,
    layer_mode: str = "final",
    target_metric: str = "median_spearman",
    output_root: Path | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, object], pd.Series | None]:
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if layer_mode == "final" or model_name.upper() == "SPECTEMP":
        return (
            run_single_model_analysis(
                model_name,
                dataset_name=dataset_name,
                output_root=output_root,
                overwrite=overwrite,
            ),
            None,
        )

    layerwise_table = collect_layerwise_metrics(
        model_name,
        dataset_name=dataset_name,
        output_root=output_root,
        overwrite=overwrite,
    )
    selected = _best_row(layerwise_table, target_metric)
    layer_index = None if pd.isna(selected["layer_index"]) else int(selected["layer_index"])
    artifact = run_single_model_analysis(
        model_name,
        dataset_name=dataset_name,
        layer_index=layer_index,
        output_root=output_root,
        overwrite=overwrite,
    )
    return artifact, selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute all dataset/model/layer regression and RSA outputs, then compile summary tables.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of model names to cache/compile (e.g., WAV2VEC2). Defaults to all.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional subset of datasets to cache/compile (e.g., NH2015 B2021). Defaults to both.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cached outputs even if they already exist.",
    )
    parser.add_argument(
        "--merge-existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "When compiling a subset (specific models or datasets), merge refreshed rows into existing "
            "compiled_results instead of replacing the whole table. "
            "Default: enabled for subset runs, disabled for full runs."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    models = [name.strip().upper() for name in args.models] if args.models else None
    datasets = args.datasets if args.datasets else ("NH2015", "B2021")
    ensure_full_cache(
        datasets=datasets,
        models=models or DEFAULT_MODELS,
        overwrite=args.overwrite,
    )
    is_subset_run = bool(args.models) or bool(args.datasets)
    merge_existing = args.merge_existing if args.merge_existing is not None else is_subset_run
    regression_pkl, rsa_pkl = write_compiled_summaries(
        datasets=datasets,
        allowed_models=models,
        merge_existing=merge_existing,
    )
    print(f"Wrote compiled regression summary: {regression_pkl}")
    print(f"Wrote compiled RSA summary: {rsa_pkl}")


if __name__ == "__main__":
    main()
