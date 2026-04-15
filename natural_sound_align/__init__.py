"""ISMIR analysis helpers for Norman 165 foundation-model comparisons."""

from .data import (
    load_brain_voxel_data,
    load_component_responses,
    load_norman_metadata,
    load_tuckute_stimulus_metadata,
    normalize_dataset_name,
)
from .embeddings import load_model_embedding


def run_all_models(*args, **kwargs):
    from .pipeline import run_all_models as _run_all_models

    return _run_all_models(*args, **kwargs)


def run_single_model_analysis(*args, **kwargs):
    from .pipeline import run_single_model_analysis as _run_single_model_analysis

    return _run_single_model_analysis(*args, **kwargs)

__all__ = [
    "load_brain_voxel_data",
    "load_component_responses",
    "load_model_embedding",
    "load_norman_metadata",
    "load_tuckute_stimulus_metadata",
    "normalize_dataset_name",
    "run_all_models",
    "run_single_model_analysis",
]
