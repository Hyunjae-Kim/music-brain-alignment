"""Layerwise model alignment against Norman 2015 response components."""

from .data import DEFAULT_COMPONENT_NAMES, load_component_bundle, select_component_indices

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

def run_single_representation_analysis(*args, **kwargs):
    from .pipeline import run_single_representation_analysis as _run_single_representation_analysis

    return _run_single_representation_analysis(*args, **kwargs)


def run_model_representations(*args, **kwargs):
    from .pipeline import run_model_representations as _run_model_representations

    return _run_model_representations(*args, **kwargs)


def run_all_models(*args, **kwargs):
    from .pipeline import run_all_models as _run_all_models

    return _run_all_models(*args, **kwargs)

__all__ = [
    "DEFAULT_COMPONENT_NAMES",
    "load_component_bundle",
    "select_component_indices",
    "DEFAULT_MODELS",
    "run_single_representation_analysis",
    "run_model_representations",
    "run_all_models",
]
