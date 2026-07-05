"""Staged HydraSim profiles for HydraSim reference water plant."""

from importlib import import_module
from typing import Any

__all__ = [
    "build_live_orchestration_plan",
    "build_runtime_artifact",
    "export_hs_bundle",
    "get_profile",
    "get_scenario",
    "profile_ids",
    "render_hs_pcap_bytes",
    "render_live_plan",
    "render_process_evolution_csv",
    "render_process_review_csv",
    "render_release_candidate_json",
    "render_release_candidate_markdown",
    "render_summary_markdown",
    "render_transcript_csv",
    "run_live_orchestration",
    "scenario_ids",
    "validate_profile",
    "build_reference_water_plant_cfd_release_candidate",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name in {"get_profile", "profile_ids"}:
        module = import_module(".profiles", __name__)
    elif name == "validate_profile":
        module = import_module(".validator", __name__)
    elif name in {"get_scenario", "scenario_ids"}:
        module = import_module(".scenarios", __name__)
    elif name == "build_runtime_artifact":
        module = import_module(".runtime", __name__)
    elif name == "export_hs_bundle":
        module = import_module(".bundle", __name__)
    elif name in {
        "build_live_orchestration_plan",
        "render_live_plan",
        "run_live_orchestration",
    }:
        module = import_module(".orchestration", __name__)
    elif name == "render_hs_pcap_bytes":
        module = import_module(".pcap", __name__)
    elif name in {
        "build_reference_water_plant_cfd_release_candidate",
        "render_release_candidate_json",
        "render_release_candidate_markdown",
    }:
        module = import_module(".release", __name__)
    elif name in {
        "render_process_evolution_csv",
        "render_process_review_csv",
        "render_summary_markdown",
        "render_transcript_csv",
    }:
        module = import_module(".render", __name__)
    else:
        module = import_module(".render", __name__)
    return getattr(module, name)
