from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from natural_sound_align.embeddings import available_layer_indices, load_model_embedding

from .data import PACKAGE_ROOT, load_component_bundle, select_component_indices
from .regression import regression_result_to_frame, run_component_regression
from .rsa import rsa_result_to_frame, run_component_rsa


DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "results"
DEFAULT_MODELS = (
    "CLAP",
    "CLAP_MUSIC",
    "CLAP_MUSIC_SPEECH",
    "MULAN",
    "MERT",
    "MERT95",
    "WAV2VEC2",
    "MUSICGEN_SMALL",
    "MUSICGEN_MEDIUM",
    "MUSICGEN_LARGE",
    "MUSICGEN_MELODY_LARGE",
)


def iter_representation_specs(
    model_name: str,
    *,
    include_final: bool = True,
) -> list[tuple[int | None, str]]:
    specs = [(layer_index, f"layer_{layer_index:02d}") for layer_index in available_layer_indices(model_name)]
    if include_final:
        specs.append((None, "final"))
    return specs


def _model_output_dir(output_root: Path, model_name: str, representation_name: str) -> Path:
    return output_root / model_name.lower() / representation_name


def _trim_representation_specs(
    specs: list[tuple[int | None, str]],
    *,
    max_representations: int | None,
) -> list[tuple[int | None, str]]:
    if max_representations is None:
        return specs
    return specs[: max(0, int(max_representations))]


def run_single_representation_analysis(
    model_name: str,
    *,
    layer_index: int | None,
    output_root: Path | None = None,
    requested_components: tuple[str, ...] | list[str] | None = None,
    overwrite: bool = False,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    seed: int = 0,
    rsa_mean_subtract: bool = True,
    rsa_with_std: bool = True,
) -> dict[str, object]:
    output_root = output_root or DEFAULT_OUTPUT_ROOT

    embedding = load_model_embedding(model_name, layer_index=layer_index, dataset_name="NH2015")
    bundle = load_component_bundle()
    component_indices = select_component_indices(bundle.component_names, requested_components)
    component_names = tuple(bundle.component_names[index] for index in component_indices)
    component_responses = bundle.responses[:, component_indices]

    if embedding.metadata["stim_name"].tolist() != bundle.metadata["stim_name"].tolist():
        raise ValueError(
            f"Stimulus order mismatch between embedding {embedding.model_name} / {embedding.representation_name} "
            "and Norman component responses."
        )

    result_dir = _model_output_dir(output_root, embedding.model_name, embedding.representation_name)
    result_dir.mkdir(parents=True, exist_ok=True)
    regression_summary_path = result_dir / "regression_summary.csv"
    rsa_summary_path = result_dir / "rsa_summary.csv"
    regression_metrics_path = result_dir / "regression_metrics.npz"
    rsa_metrics_path = result_dir / "rsa_metrics.npz"

    if not overwrite and all(
        path.exists()
        for path in (regression_summary_path, rsa_summary_path, regression_metrics_path, rsa_metrics_path)
    ):
        regression_summary = pd.read_csv(regression_summary_path)
        rsa_summary = pd.read_csv(rsa_summary_path)
        return {
            "embedding": embedding,
            "component_names": component_names,
            "regression_summary": regression_summary,
            "rsa_summary": rsa_summary,
            "result_dir": result_dir,
        }

    regression = run_component_regression(
        embedding.matrix,
        component_responses,
        component_names,
        n_splits=n_splits,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    regression_summary = regression_result_to_frame(regression)
    regression_summary.insert(0, "representation_name", embedding.representation_name)
    regression_summary.insert(0, "model_name", embedding.model_name)
    regression_summary.insert(2, "layer_index", embedding.layer_index)

    rsa = run_component_rsa(
        embedding.matrix,
        component_responses,
        component_names,
        n_splits=n_splits,
        n_train=n_train,
        n_test=n_test,
        random_seed=seed,
        mean_subtract=rsa_mean_subtract,
        with_std=rsa_with_std,
    )
    rsa_summary = rsa_result_to_frame(rsa)
    rsa_summary.insert(0, "representation_name", embedding.representation_name)
    rsa_summary.insert(0, "model_name", embedding.model_name)
    rsa_summary.insert(2, "layer_index", embedding.layer_index)

    regression_summary.to_csv(regression_summary_path, index=False)
    rsa_summary.to_csv(rsa_summary_path, index=False)
    np.savez_compressed(
        regression_metrics_path,
        component_names=np.asarray(component_names, dtype=str),
        split_r2=regression.split_r2,
        split_train_r2=regression.split_train_r2,
        split_alpha=regression.split_alpha,
    )
    np.savez_compressed(
        rsa_metrics_path,
        component_names=np.asarray(component_names, dtype=str),
        split_scores=rsa.split_scores,
        model_rdm=rsa.model_rdm,
        component_rdms=rsa.component_rdms,
    )

    return {
        "embedding": embedding,
        "component_names": component_names,
        "regression_summary": regression_summary,
        "rsa_summary": rsa_summary,
        "result_dir": result_dir,
    }


def run_model_representations(
    model_name: str,
    *,
    output_root: Path | None = None,
    include_final: bool = True,
    max_representations: int | None = None,
    requested_components: tuple[str, ...] | list[str] | None = None,
    overwrite: bool = False,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    seed: int = 0,
    rsa_mean_subtract: bool = True,
    rsa_with_std: bool = True,
) -> dict[str, object]:
    specs = _trim_representation_specs(
        iter_representation_specs(model_name, include_final=include_final),
        max_representations=max_representations,
    )
    artifacts = []
    regression_rows = []
    rsa_rows = []
    for layer_index, _ in specs:
        artifact = run_single_representation_analysis(
            model_name,
            layer_index=layer_index,
            output_root=output_root,
            requested_components=requested_components,
            overwrite=overwrite,
            n_splits=n_splits,
            n_train=n_train,
            n_test=n_test,
            seed=seed,
            rsa_mean_subtract=rsa_mean_subtract,
            rsa_with_std=rsa_with_std,
        )
        artifacts.append(artifact)
        regression_rows.append(artifact["regression_summary"])
        rsa_rows.append(artifact["rsa_summary"])

    regression_summary = pd.concat(regression_rows, ignore_index=True) if regression_rows else pd.DataFrame()
    rsa_summary = pd.concat(rsa_rows, ignore_index=True) if rsa_rows else pd.DataFrame()
    return {
        "model_name": model_name.strip().upper(),
        "artifacts": artifacts,
        "regression_summary": regression_summary,
        "rsa_summary": rsa_summary,
    }


def run_all_models(
    model_names: tuple[str, ...] | list[str] = DEFAULT_MODELS,
    *,
    output_root: Path | None = None,
    include_final: bool = True,
    max_representations: int | None = None,
    requested_components: tuple[str, ...] | list[str] | None = None,
    overwrite: bool = False,
    n_splits: int = 10,
    n_train: int = 83,
    n_test: int = 82,
    seed: int = 0,
    rsa_mean_subtract: bool = True,
    rsa_with_std: bool = True,
) -> dict[str, object]:
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    regression_rows = []
    rsa_rows = []
    for model_name in model_names:
        artifact = run_model_representations(
            model_name,
            output_root=output_root,
            include_final=include_final,
            max_representations=max_representations,
            requested_components=requested_components,
            overwrite=overwrite,
            n_splits=n_splits,
            n_train=n_train,
            n_test=n_test,
            seed=seed,
            rsa_mean_subtract=rsa_mean_subtract,
            rsa_with_std=rsa_with_std,
        )
        artifacts[model_name.strip().upper()] = artifact
        if not artifact["regression_summary"].empty:
            regression_rows.append(artifact["regression_summary"])
        if not artifact["rsa_summary"].empty:
            rsa_rows.append(artifact["rsa_summary"])

    regression_summary = pd.concat(regression_rows, ignore_index=True) if regression_rows else pd.DataFrame()
    rsa_summary = pd.concat(rsa_rows, ignore_index=True) if rsa_rows else pd.DataFrame()

    if not regression_summary.empty:
        regression_summary.to_csv(output_root / "regression_layerwise_summary.csv", index=False)
    if not rsa_summary.empty:
        rsa_summary.to_csv(output_root / "rsa_layerwise_summary.csv", index=False)

    return {
        "artifacts": artifacts,
        "regression_summary": regression_summary,
        "rsa_summary": rsa_summary,
        "output_root": output_root,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run layerwise alignment against Norman response components.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Models to analyze.")
    parser.add_argument(
        "--components",
        nargs="+",
        default=None,
        help="Optional subset of components. Default: all six components.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where component-alignment outputs will be stored.",
    )
    parser.add_argument(
        "--max-representations",
        type=int,
        default=None,
        help="Optional limit on the number of representations per model, useful for smoke tests.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="If set, analyze only explicit layer representations and omit the final representation.",
    )
    parser.add_argument("--n-splits", type=int, default=10, help="Number of random train/test splits.")
    parser.add_argument("--n-train", type=int, default=83, help="Training sounds per split.")
    parser.add_argument("--n-test", type=int, default=82, help="Test sounds per split.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split generation.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_all_models(
        model_names=args.models,
        output_root=Path(args.output_root),
        include_final=not args.skip_final,
        max_representations=args.max_representations,
        requested_components=args.components,
        overwrite=args.overwrite,
        n_splits=args.n_splits,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )
    if not result["regression_summary"].empty:
        print("Regression summary")
        print(result["regression_summary"].to_string(index=False))
        print()
    if not result["rsa_summary"].empty:
        print("RSA summary")
        print(result["rsa_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
