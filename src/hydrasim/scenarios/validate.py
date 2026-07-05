"""Command-line scenario validator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .loader import load_scenario_json
from .profiles import SCENARIO_ALIASES, get_scenario, scenario_ids
from .validator import validate_scenario


def _load(identifier: str, custom_json: Path | None):
    if custom_json is not None:
        return load_scenario_json(custom_json)
    return get_scenario(identifier)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a HydraSim scenario")
    parser.add_argument(
        "scenario", choices=scenario_ids() + tuple(SCENARIO_ALIASES) + ("custom",)
    )
    parser.add_argument("--custom-json", type=Path)
    args = parser.parse_args(argv)

    scenario = _load(args.scenario, args.custom_json)
    errors = validate_scenario(scenario)
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    print(f"valid: {scenario.scenario_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
