from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PACKAGE_ROOT, load_brain_voxel_data, normalize_dataset_name
from .embeddings import ensure_model_embeddings, load_model_embedding
from .regression import (
    RegressionResult,
    aggregate_subject_cv_splits,
    regression_result_to_frame,
    run_noise_corrected_regression,
    summarize_subject_aggregation,
    summarize_regression_result,
)
from .rsa import RSAResult, run_rsa, summarize_rsa_result


DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "embeddings"
DEFAULT_MODELS = (
    "CLAP",
    "CLAP_MUSIC",
    "CLAP_MUSIC_SPEECH",
    "MULAN",
    "MERT",
    "MERT95",
    "MUSICGEN_SMALL",
    "MUSICGEN_MEDIUM",
    "MUSICGEN_LARGE",
    "MUSICGEN_MELODY_LARGE",
    "SPECTEMP",
)
DEFAULT_RSA_MEAN_SUBTRACT = True
DEFAULT_RSA_WITH_STD = True


def _dataset_output_root(output_root: Path, dataset_name: str) -> Path:
    return output_root / normalize_dataset_name(dataset_name).lower()


def _model_output_dir(output_root: Path, dataset_name: str, model_name: str, representation_name: str) -> Path:
    dataset_root = _dataset_output_root(output_root, dataset_name)
    return dataset_root / model_name.lower() / representation_name


def _load_cached_analysis(
    model_name: str,
    *,
    dataset_name: str,
    layer_index: int | None,
    output_root: Path,
) -> dict[str, object] | None:
    try:
        embedding = load_model_embedding(model_name, layer_index=layer_index, dataset_name=dataset_name)
    except FileNotFoundError:
        return None
    brain = load_brain_voxel_data(dataset_name=dataset_name, require_complete_repetitions=True)
    result_dir = _model_output_dir(output_root, dataset_name, embedding.model_name, embedding.representation_name)

    regression_summary_path = result_dir / "regression_summary.json"
    regression_voxelwise_path = result_dir / "regression_voxelwise.csv"
    regression_metrics_path = result_dir / "regression_split_metrics.npz"
    regression_subject_path = result_dir / "regression_subject_cv10.csv"
    regression_subject_stats_path = result_dir / "regression_subject_cv10_stats.csv"
    rsa_summary_path = result_dir / "rsa_summary.json"
    rsa_metrics_path = result_dir / "rsa_metrics.npz"

    required_paths = [
        regression_summary_path,
        regression_voxelwise_path,
        regression_metrics_path,
        regression_subject_path,
        regression_subject_stats_path,
        rsa_summary_path,
        rsa_metrics_path,
    ]
    if not all(path.exists() for path in required_paths):
        return None

    regression_summary = json.loads(regression_summary_path.read_text())
    regression_frame = pd.read_csv(regression_voxelwise_path)
    regression_subject_frame = pd.read_csv(regression_subject_path)
    regression_subject_stats_frame = pd.read_csv(regression_subject_stats_path)
    with np.load(regression_metrics_path, allow_pickle=True) as npz_file:
        regression = RegressionResult(
            split_r2=np.asarray(npz_file["split_r2"], dtype=np.float32),
            split_corrected_r2=np.asarray(npz_file["split_corrected_r2"], dtype=np.float32),
            split_train_r2=np.asarray(npz_file["split_train_r2"], dtype=np.float32),
            split_alpha=np.asarray(npz_file["split_alpha"], dtype=np.float64),
            voxel_ids=np.asarray(npz_file["voxel_ids"], dtype=int),
            splits=tuple(),
            source_dim=int(embedding.matrix.shape[1]),
        )

    rsa_summary = json.loads(rsa_summary_path.read_text())
    with np.load(rsa_metrics_path, allow_pickle=True) as npz_file:
        scores_by_repetition = np.asarray(npz_file["scores_by_repetition"], dtype=np.float64)
        participant_ids = (
            np.asarray(npz_file["participant_ids"], dtype=int)
            if "participant_ids" in npz_file.files
            else np.arange(scores_by_repetition.shape[0], dtype=int)
        )
        split_scores = (
            np.asarray(npz_file["split_scores"], dtype=np.float64)
            if "split_scores" in npz_file.files
            else scores_by_repetition[:, None]
        )
        leave_one_out_neural_scores = (
            np.asarray(npz_file["leave_one_out_neural_scores"], dtype=np.float64)
            if "leave_one_out_neural_scores" in npz_file.files
            else np.full_like(split_scores, np.nan)
        )
        rsa = RSAResult(
            model_name=embedding.model_name,
            representation_name=embedding.representation_name,
            scores_by_repetition=scores_by_repetition,
            mean_spearman=float(np.mean(scores_by_repetition)),
            model_correlation_matrix=np.asarray(npz_file["model_correlation_matrix"], dtype=np.float64),
            human_correlation_matrices=np.asarray(npz_file["human_correlation_matrices"], dtype=np.float64),
            mean_human_correlation_matrix=np.asarray(npz_file["mean_human_correlation_matrix"], dtype=np.float64),
            participant_ids=participant_ids,
            split_scores=split_scores,
            leave_one_out_neural_scores=leave_one_out_neural_scores,
        )

    return {
        "embedding": embedding,
        "brain": brain,
        "regression": regression,
        "regression_summary": regression_summary,
        "regression_frame": regression_frame,
        "regression_subject_frame": regression_subject_frame,
        "regression_subject_stats_frame": regression_subject_stats_frame,
        "rsa": rsa,
        "rsa_summary": rsa_summary,
        "result_dir": result_dir,
    }


def run_single_model_analysis(
    model_name: str,
    *,
    dataset_name: str = "NH2015",
    layer_index: int | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
    ensure_embeddings: bool = False,
    device: str = "cuda",
) -> dict[str, object]:
    dataset_name = normalize_dataset_name(dataset_name)
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    _dataset_output_root(output_root, dataset_name).mkdir(parents=True, exist_ok=True)

    if not overwrite:
        cached = _load_cached_analysis(
            model_name,
            dataset_name=dataset_name,
            layer_index=layer_index,
            output_root=output_root,
        )
        if cached is not None:
            return cached

    if ensure_embeddings and model_name.strip().upper() != "SPECTEMP":
        ensure_model_embeddings(model_name, overwrite=overwrite, device=device)

    embedding = load_model_embedding(model_name, layer_index=layer_index, dataset_name=dataset_name)
    brain = load_brain_voxel_data(dataset_name=dataset_name, require_complete_repetitions=True)

    result_dir = _model_output_dir(output_root, dataset_name, embedding.model_name, embedding.representation_name)
    result_dir.mkdir(parents=True, exist_ok=True)

    regression = run_noise_corrected_regression(
        embedding.matrix,
        brain.responses,
        brain.voxel_ids,
    )
    subject_aggregation = aggregate_subject_cv_splits(
        regression.split_corrected_r2,
        brain.voxel_metadata,
        value_column="median_r2_test_c",
        nit=10,
        collapse_over_splits="median",
        clip_upper=1.0,
    )
    regression_summary = summarize_regression_result(
        regression,
        model_name=embedding.model_name,
        representation_name=embedding.representation_name,
    ) | summarize_subject_aggregation(
        subject_aggregation,
        model_name=embedding.model_name,
        representation_name=embedding.representation_name,
    ) | {"dataset_name": dataset_name}
    regression_frame = regression_result_to_frame(regression)
    regression_subject_frame = subject_aggregation.subject_scores
    regression_subject_stats_frame = subject_aggregation.iteration_scores

    rsa = run_rsa(
        embedding.matrix,
        brain.responses,
        brain.metadata,
        voxel_metadata=brain.voxel_metadata,
        model_name=embedding.model_name,
        representation_name=embedding.representation_name,
        mean_subtract=DEFAULT_RSA_MEAN_SUBTRACT,
        with_std=DEFAULT_RSA_WITH_STD,
    )
    rsa_summary = summarize_rsa_result(rsa) | {"dataset_name": dataset_name}

    regression_frame.to_csv(result_dir / "regression_voxelwise.csv", index=False)
    regression_subject_frame.to_csv(result_dir / "regression_subject_cv10.csv", index=False)
    regression_subject_stats_frame.to_csv(result_dir / "regression_subject_cv10_stats.csv", index=False)
    with open(result_dir / "regression_summary.json", "w", encoding="utf-8") as file:
        json.dump(regression_summary, file, indent=2)
    np.savez_compressed(
        result_dir / "regression_split_metrics.npz",
        split_r2=regression.split_r2,
        split_corrected_r2=regression.split_corrected_r2,
        split_train_r2=regression.split_train_r2,
        split_alpha=regression.split_alpha,
        voxel_ids=regression.voxel_ids,
    )

    with open(result_dir / "rsa_summary.json", "w", encoding="utf-8") as file:
        json.dump(rsa_summary, file, indent=2)
    np.savez_compressed(
        result_dir / "rsa_metrics.npz",
        scores_by_repetition=rsa.scores_by_repetition,
        model_correlation_matrix=rsa.model_correlation_matrix,
        human_correlation_matrices=rsa.human_correlation_matrices,
        mean_human_correlation_matrix=rsa.mean_human_correlation_matrix,
        participant_ids=rsa.participant_ids,
        split_scores=rsa.split_scores,
        leave_one_out_neural_scores=rsa.leave_one_out_neural_scores,
    )

    return {
        "embedding": embedding,
        "brain": brain,
        "regression": regression,
        "regression_summary": regression_summary,
        "regression_frame": regression_frame,
        "regression_subject_frame": regression_subject_frame,
        "regression_subject_stats_frame": regression_subject_stats_frame,
        "rsa": rsa,
        "rsa_summary": rsa_summary,
        "result_dir": result_dir,
    }


def run_all_models(
    model_names: tuple[str, ...] | list[str] = DEFAULT_MODELS,
    *,
    dataset_name: str = "NH2015",
    output_root: Path | None = None,
    overwrite: bool = False,
    ensure_embeddings: bool = False,
    device: str = "cuda",
) -> dict[str, object]:
    dataset_name = normalize_dataset_name(dataset_name)
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    dataset_root = _dataset_output_root(output_root, dataset_name)
    dataset_root.mkdir(parents=True, exist_ok=True)

    regression_rows = []
    rsa_rows = []
    artifacts: dict[str, object] = {}
    for model_name in model_names:
        artifact = run_single_model_analysis(
            model_name,
            dataset_name=dataset_name,
            output_root=output_root,
            overwrite=overwrite,
            ensure_embeddings=ensure_embeddings,
            device=device,
        )
        artifacts[model_name.upper()] = artifact
        regression_rows.append(artifact["regression_summary"])
        rsa_rows.append(artifact["rsa_summary"])

    regression_summary = pd.DataFrame(regression_rows).sort_values("cv_subject_mean", ascending=False)
    rsa_summary = pd.DataFrame(rsa_rows).sort_values("mean_spearman", ascending=False)

    regression_summary.to_csv(dataset_root / "regression_summary.csv", index=False)
    rsa_summary.to_csv(dataset_root / "rsa_summary.csv", index=False)

    return {
        "artifacts": artifacts,
        "regression_summary": regression_summary,
        "rsa_summary": rsa_summary,
        "output_root": dataset_root,
        "dataset_name": dataset_name,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Norman 165 model-vs-fMRI regression and RSA analyses.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="NH2015",
        help="Dataset to analyze. Supported: NH2015, N2015, B2021",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=(
            "Model names to analyze. Supported: "
            "CLAP CLAP_MUSIC CLAP_MUSIC_SPEECH MULAN MERT MERT95 "
            "MUSICGEN_SMALL MUSICGEN_MEDIUM MUSICGEN_LARGE MUSICGEN_MELODY_LARGE "
            "SPECTEMP"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where analysis outputs will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if existing files are present.",
    )
    parser.add_argument(
        "--ensure-embeddings",
        action="store_true",
        help="If set, rerun the embedding extraction scripts when outputs are missing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string passed to embedding extraction when --ensure-embeddings is used.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_all_models(
        model_names=args.models,
        dataset_name=args.dataset,
        output_root=Path(args.output_root),
        overwrite=args.overwrite,
        ensure_embeddings=args.ensure_embeddings,
        device=args.device,
    )
    print(result["regression_summary"].to_string(index=False))
    print()
    print(result["rsa_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
