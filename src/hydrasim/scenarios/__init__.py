"""Scenario profiles and deterministic transcript helpers for HydraSim."""

from importlib import import_module
from typing import Any

__all__ = [
    "FieldPoint",
    "ModbusTransaction",
    "NetworkNode",
    "ScenarioProfile",
    "get_scenario",
    "export_lab_bundle",
    "load_scenario_json",
    "render_markdown_summary",
    "render_pcap_bytes",
    "render_results_csv",
    "render_transcript_csv",
    "run_scenario",
    "run_scenario_with_clients",
    "scenario_ids",
    "validate_scenario",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    if name in {
        "FieldPoint",
        "ModbusTransaction",
        "NetworkNode",
        "ScenarioProfile",
    }:
        module = import_module(".models", __name__)
    elif name == "export_lab_bundle":
        module = import_module(".bundle", __name__)
    elif name in {"get_scenario", "scenario_ids"}:
        module = import_module(".profiles", __name__)
    elif name == "load_scenario_json":
        module = import_module(".loader", __name__)
    elif name in {"render_markdown_summary", "render_transcript_csv"}:
        module = import_module(".render", __name__)
    elif name == "render_pcap_bytes":
        module = import_module(".pcap", __name__)
    elif name == "validate_scenario":
        module = import_module(".validator", __name__)
    else:
        module = import_module(".executor", __name__)
    return getattr(module, name)
