from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from natural_sound_align.embeddings import available_layer_indices

from .data import PACKAGE_ROOT
from .pipeline import DEFAULT_MODELS, DEFAULT_OUTPUT_ROOT


COMPILED_ROOT = PACKAGE_ROOT / "compiled_results"


def _layer_depth(model_name: str, representation_name: str) -> int:
    if representation_name == "final":
        layers = available_layer_indices(model_name)
        return 0 if not layers else max(layers) + 1
    if representation_name.startswith("layer_"):
        return int(representation_name.split("_", 1)[1])
    return 0


def collect_summaries(
    *,
    output_root: Path | None = None,
    allowed_models: tuple[str, ...] | list[str] = DEFAULT_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    allowed = {model_name.strip().upper() for model_name in allowed_models}
    regression_rows = []
    rsa_rows = []

    for model_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        if model_dir.name.upper() not in allowed:
            continue
        for representation_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            regression_path = representation_dir / "regression_summary.csv"
            rsa_path = representation_dir / "rsa_summary.csv"
            if regression_path.exists():
                frame = pd.read_csv(regression_path)
                frame["layer_depth"] = _layer_depth(model_dir.name, representation_dir.name)
                frame["is_final"] = representation_dir.name == "final"
                frame["result_dir"] = str(representation_dir)
                regression_rows.append(frame)
            if rsa_path.exists():
                frame = pd.read_csv(rsa_path)
                frame["layer_depth"] = _layer_depth(model_dir.name, representation_dir.name)
                frame["is_final"] = representation_dir.name == "final"
                frame["result_dir"] = str(representation_dir)
                rsa_rows.append(frame)

    regression_df = pd.concat(regression_rows, ignore_index=True) if regression_rows else pd.DataFrame()
    rsa_df = pd.concat(rsa_rows, ignore_index=True) if rsa_rows else pd.DataFrame()
    return regression_df, rsa_df


def write_compiled_summaries(
    *,
    output_root: Path | None = None,
    compiled_root: Path | None = None,
    allowed_models: tuple[str, ...] | list[str] = DEFAULT_MODELS,
) -> tuple[Path, Path]:
    compiled_root = compiled_root or COMPILED_ROOT
    compiled_root.mkdir(parents=True, exist_ok=True)
    regression_df, rsa_df = collect_summaries(output_root=output_root, allowed_models=allowed_models)

    regression_csv = compiled_root / "regression_layerwise_summary.csv"
    rsa_csv = compiled_root / "rsa_layerwise_summary.csv"
    regression_pkl = compiled_root / "regression_layerwise_summary.pkl"
    rsa_pkl = compiled_root / "rsa_layerwise_summary.pkl"

    regression_df.to_csv(regression_csv, index=False)
    rsa_df.to_csv(rsa_csv, index=False)
    regression_df.to_pickle(regression_pkl)
    rsa_df.to_pickle(rsa_pkl)
    return regression_pkl, rsa_pkl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile cached natural-component alignment summaries.")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--compiled-root", type=str, default=str(COMPILED_ROOT))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    regression_pkl, rsa_pkl = write_compiled_summaries(
        output_root=Path(args.output_root),
        compiled_root=Path(args.compiled_root),
        allowed_models=args.models,
    )
    print(f"Wrote {regression_pkl}")
    print(f"Wrote {rsa_pkl}")


if __name__ == "__main__":
    main()
