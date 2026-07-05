# HydraSim

[![Quality](https://github.com/Guivernoir/HydraSim/actions/workflows/quality.yml/badge.svg)](https://github.com/Guivernoir/HydraSim/actions/workflows/quality.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.13%20%7C%203.14-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Code files](https://img.shields.io/badge/code%20files-%3C%3D500%20lines-brightgreen)

HydraSim is a Python water-treatment process simulator for control-system
integration, Modbus testing, synthetic plant traffic, and bounded CFD/digital
twin experiments.

It gives you a local process endpoint that behaves like a small field-facing
water plant unit: reactor physics evolve over time, actuators change process
boundaries, sensors report delayed/noisy/faultable measurements, and Modbus
registers expose command and feedback surfaces for PLC/SCADA-style clients.

HydraSim is simulation and test infrastructure. It is **not certified design authority**,
commissioning evidence, safety validation, or real-plant validation, and it is
not proof of plant equivalence without separate calibration and external
validation.

## Repository Status

HydraSim is the project root and the public GitHub surface is intentionally
small: source code, tests, examples, CI policy, package metadata, license,
security policy, and this README. Internal planning notes stay in ignored
`.private/docs` files and are not part of the public repository.

The package targets Python 3.11 and newer, with CI coverage for Python 3.11,
3.13, and 3.14. CI is also the repository quality contract: formatting, lint,
types, dependency checks, syntax compilation, deterministic project checks, the
500-line code-file limit, and the full unit/Modbus test suite must all pass.

## Why HydraSim

Control-system integration work often needs a repeatable process target before
plant hardware, PLC logic, or historian infrastructure is available. HydraSim
fills that gap by giving local clients deterministic process behavior,
plant-style register surfaces, generated evidence artifacts, and clear limits on
what is synthetic versus externally validated.

## What It Does

- Simulates multi-zone reactor physics: mixing, advection, pH/chlorine
  chemistry, ammonia/chloramine behavior, demand, and temperature.
- Models realistic sensors and actuators: delay, noise, drift, warm-up,
  saturation, faults, valves, and dosing pumps.
- Exposes a Modbus TCP process endpoint with plant-style command and feedback
  registers.
- Generates deterministic Modbus scenarios, transcripts, PCAPs, and lab
  bundles for repeatable local testing.
- Provides staged Reference Water Plant profiles for offline export, selected
  area runs, and live-plan generation.
- Includes bounded CFD/digital-twin primitives and evidence gates that separate
  implementation verification from real-plant validation.
- Enforces repository quality with formatting, lint, type checks, tests,
  public README coverage checks, and a no-exceptions 500-line code-file limit.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,modbus]"
python -m hydrasim --no-modbus --duration 10 --dt 1
```

Run the full local gate:

```bash
python -m pip check
python -m black --check src tests tools
python -m ruff check src tools tests
python -m mypy src/hydrasim
python -m compileall -q src tests tools
python tools/check_project_quality.py
python -m unittest discover -s tests -v
```

## Common Commands

Start a Modbus process endpoint:

```bash
python -m hydrasim --host 127.0.0.1 --port 5020
```

Run a built-in scenario against that endpoint:

```bash
hs-run-scenario water-treatment-normal --mode live --host 127.0.0.1 --port 5020
```

Start the simulator and replay a scenario from one command:

```bash
hs-sim --host 127.0.0.1 --port 5020 --scenario water-treatment-normal
```

Export deterministic scenario evidence:

```bash
hs-run-scenario water-treatment-smart-field --mode transcript --format markdown
hs-run-scenario water-treatment-noisy-network --mode transcript --format pcap --output noisy.pcap
hs-export-lab-bundle water-treatment-smart-field ./hydrasim-smart-field-bundle
```

Work with the staged Reference Water Plant profiles:

```bash
hs-plant list-profiles
hs-plant validate-profile reference-water-plant
hs-plant run reference-water-plant --scenario HS-WTP-002 --area disinfection --stage full-cell --format markdown
hs-plant export-bundle reference-water-plant HS-WTP-002 ./reference-water-bundle --area all --stage offline-export
hs-plant launch-live reference-water-plant --scenario HS-WTP-002 --area disinfection --stage full-cell --dry-run
```

Run or validate a custom JSON scenario:

```bash
hs-run-scenario custom --custom-json examples/custom_scenario_template.json --mode transcript
hs-validate-scenario custom --custom-json examples/custom_scenario_template.json
hs-sim --scenario custom --scenario-custom-json examples/custom_scenario_template.json
```

## Public Surface

HydraSim keeps internal planning notes private. The public repository exposes
the simulator, tests, examples, CI policy, and this README.

Built-in MVP scenario IDs:
`MVP-MB-HYDRA-002`, `MVP-MB-HYDRA-003`, `MVP-MB-HYDRA-004`,
`MVP-MB-HYDRA-005`, `MVP-MB-HYDRA-006`, `MVP-MB-HYDRA-007`,
`MVP-MB-HYDRA-008`, and `MVP-MB-HYDRA-009`.

Reference Water Plant profiles:
`single-stage-legacy`, `field-device-lab`, `controller-cell`,
`supervisory-lab`, and `reference-water-plant`.

Reference Water Plant scenario IDs:
`HS-WTP-001`, `HS-WTP-002`, `HS-WTP-003`, `HS-WTP-004`, `HS-WTP-005`,
`HS-WTP-006`, `HS-WTP-007`, `HS-WTP-008`, `HS-WTP-009`, `HS-WTP-010`,
`HS-WTP-011`, and `HS-WTP-012`.

Reference plant areas:
`intake`, `dosing`, `clarification`, `filtration`, `disinfection`, and
`storage-pumping`.

CFD/digital-twin public evidence surfaces include the Runtime Performance Gate,
Digital-Twin Validation Gate, External Review And Calibration Evidence Gate,
CFD Lab Bundle v2, and Reference Water Plant CFD release-candidate output. CFD
evidence is synthetic unless separately calibrated and externally validated.

## Architecture

The runtime loop keeps controller intent outside the physics model:

1. Read Modbus holding registers and coils.
2. Apply commands to actuator models.
3. Map actuator outputs to reactor boundary flows.
4. Step reactor physics.
5. Read sensors from reactor state.
6. Publish sensor values and status to Modbus input registers and discrete inputs.
7. Poll maintenance registers and dispatch any pending maintenance action.

The main packages are:

- `src/hydrasim/core`: reactor physics, chemistry, transport, and spatial models.
- `src/hydrasim/actuators`: control valves and dosing pumps.
- `src/hydrasim/sensors`: sensor models and suite factory.
- `src/hydrasim/modbus`: register map, encoding, and Modbus TCP server.
- `src/hydrasim/maintenance`: remote recalibration and hardware replacement actions.
- `src/hydrasim/scenarios`: deterministic Modbus scenario library and runner.
- `src/hydrasim/plant`: staged Reference Water Plant profiles, artifacts, and CLI.
- `src/hydrasim/hydraulics`: bounded CFD/digital-twin primitives.

## Quality Standard

HydraSim CI installs `.[dev,modbus]` and enforces the same gate intended for
local development:

- Black formatting on `src`, `tests`, and `tools`.
- Ruff linting on `src`, `tests`, and `tools`.
- Mypy type checking for `src/hydrasim`.
- Dependency consistency checks through `pip check`.
- Syntax compilation checks through `compileall`.
- Project quality policy checks, including public README coverage, deterministic
  artifact checks, Modbus dependency checks, folder density, and the hard
  500-line limit for every Python code file.
- Full unit and live Modbus end-to-end test discovery.

The quality gate has no oversized-module allowlist. If a Python file grows past
500 lines, CI fails and the code should be split before merging.

## License

MIT. See [LICENSE](LICENSE).

## Security

HydraSim is simulation-only infrastructure. Do not use it to test systems you do
not own or administer. Report vulnerabilities and sensitive safety concerns
through the private process in [SECURITY.md](SECURITY.md).
