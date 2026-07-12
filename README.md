# HydraSim

[![Quality](https://github.com/Guivernoir/HydraSim/actions/workflows/quality.yml/badge.svg)](https://github.com/Guivernoir/HydraSim/actions/workflows/quality.yml)
[![Release](https://github.com/Guivernoir/HydraSim/actions/workflows/release.yml/badge.svg)](https://github.com/Guivernoir/HydraSim/actions/workflows/release.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)
![Node](https://img.shields.io/badge/node-24-green)
![Rust](https://img.shields.io/badge/rust-stable-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Code files](https://img.shields.io/badge/code%20files-%3C%3D500%20lines-brightgreen)

HydraSim is a water-treatment simulator with a strict split between physical
simulation, runtime systems, and operator interface.

Python owns the process model: reactor physics, actuators, and sensors. Rust
owns runtime systems: HMI API, PCS validation/interlocks, Modbus TCP, and the
Python simulation worker boundary. Svelte owns the HMI surface only: display
state and submit operator intent.

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
3.12, 3.13, and 3.14. The HMI targets Node 24 and current SvelteKit tooling.
The Rust runtime targets stable Rust. CI is also the repository quality
contract: formatting, lint, types, dependency checks, syntax compilation,
deterministic project checks, the 500-line code-file limit, and the full
Python/HMI/Rust runtime test suites must all pass.

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
- Exposes plant-style command and feedback surfaces through the Rust runtime.
- Generates deterministic Modbus scenarios, transcripts, PCAPs, and lab
  bundles for repeatable local testing.
- Provides staged Reference Water Plant profiles for offline export, selected
  area runs, and live-plan generation.
- Ships a SvelteKit HMI dashboard and Rust runtime for simulation-only process
  visibility, individual signal charts, alarms, scenario selection, setpoints,
  coils, and Modbus.
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

Expected startup output includes the initialized reactor, actuator suite, sensor
suite, and clean shutdown path:

```text
HYDRASIM REACTOR SIMULATION
[PHASE 1] Initializing physics engine...
Reactor initialized: 5 zones, V=1000.0L
[PHASE 5] Skipping Modbus (--no-modbus)
[PHASE 6] Starting simulation loop...
Simulation stopped cleanly
```

Run the full local gate:

```bash
python -m pip check
python -m black --check src tests tools
python -m ruff check src tools tests
python -m mypy src/hydrasim
python -m compileall -q src tests tools
python tools/check_project_quality.py
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
python -m coverage xml
```

Run the HMI locally:

```bash
cd hmi
npm ci
npm run dev
```

Run the Rust runtime as the integrated HMI host. It launches the Python
simulation worker over stdio, serves the HMI and API on `127.0.0.1:8088`, and
keeps Modbus TCP disabled unless you explicitly enable it.

```bash
cd hmi
npm run build
cd ../runtime
HS_RUNTIME_PYTHON=../.venv/bin/python cargo run
```

Open `http://127.0.0.1:8088`. The same origin serves the Svelte HMI, static
assets, and `/api/*` runtime endpoints.

Enable the Modbus TCP integration port only when testing a PLC or Modbus
client:

```bash
cd runtime
HS_RUNTIME_PYTHON=../.venv/bin/python HS_RUNTIME_MODBUS_ENABLED=true cargo run
```

Useful runtime probes from another shell:

```bash
curl http://127.0.0.1:8088/api/health
curl http://127.0.0.1:8088/api/snapshot
```

Run the HMI and Rust runtime quality gates:

```bash
cd hmi
npm audit
npm run lint
npm run check
npm run test
npm run build
cd ../runtime
cargo fmt --check
cargo check --locked
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```

The integrated HMI smoke path is:

1. Build the HMI with `npm run build`.
2. Start Rust with `HS_RUNTIME_PYTHON=../.venv/bin/python cargo run`.
3. Open `http://127.0.0.1:8088`.
4. Confirm `/api/health` reports `runtime: "rust"` and `modbusEnabled: false`.
5. For Modbus work, restart with `HS_RUNTIME_MODBUS_ENABLED=true` and confirm
   `127.0.0.1:5502` accepts Modbus TCP reads.

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
the simulator, tests, examples, CI policy, package metadata, license, security
policy, contribution guide, changelog, and this README.

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

HydraSim uses three ownership boundaries:

1. `hmi`: SvelteKit operator interface. It renders state and sends operator
   intent over HTTP. It does not contain process, PCS, or protocol logic.
2. `runtime`: Rust runtime. It owns the HMI API, PCS validation/interlocks,
   Modbus TCP server, command state, and the Python worker lifecycle.
3. `src/hydrasim`: Python process simulation. It owns physics, sensors,
   actuators, physical scenarios, and the stdio simulation worker contract.

The live HMI path is:

```text
Svelte HMI -> Rust HTTP API -> Rust PCS/runtime -> Python simulation worker
                                      |
                                      +-> Rust Modbus TCP server
```

The main packages are:

- `src/hydrasim/core`: reactor physics, chemistry, transport, and spatial models.
- `src/hydrasim/actuators`: control valves and dosing pumps.
- `src/hydrasim/sensors`: sensor models and suite factory.
- `src/hydrasim/modbus`: legacy scenario/register tooling used by existing
  Python test surfaces.
- `src/hydrasim/maintenance`: remote recalibration and hardware replacement actions.
- `src/hydrasim/scenarios`: deterministic Modbus scenario library and runner.
- `src/hydrasim/plant`: staged Reference Water Plant profiles, artifacts, and CLI.
- `src/hydrasim/hydraulics`: bounded CFD/digital-twin primitives.
- `hmi`: SvelteKit simulation HMI with static-build output and no direct
  browser-side process, PCS, or Modbus control.
- `runtime`: Rust HTTP, PCS, Modbus, and Python-worker runtime.

## Quality Standard

HydraSim CI installs `.[dev,modbus]` and enforces the same gate intended for
local development. HMI CI installs from `hmi/package-lock.json`, and Rust CI
uses the root Cargo workspace lockfile.

- Black formatting on `src`, `tests`, and `tools`.
- Ruff linting on `src`, `tests`, and `tools`.
- Mypy type checking for `src/hydrasim`.
- Dependency consistency checks through `pip check`.
- Syntax compilation checks through `compileall`.
- Project quality policy checks, including public README coverage, deterministic
  artifact checks, Modbus dependency checks, folder density, and the hard
  500-line limit for Python, Rust, Svelte, TypeScript, JavaScript, and CSS code
  files.
- Coverage-enforced unit and live Modbus end-to-end test discovery, with
  per-Python-version XML artifacts uploaded by CI.
- HMI dependency audit, ESLint, Prettier, Svelte type checking, Vitest unit
  tests, and production SvelteKit build.
- Rust runtime formatting, locked dependency check, unit tests, and clippy with
  warnings denied.

The quality gate has no oversized-module allowlist. If a Python file grows past
500 lines, or an HMI/runtime code file grows past 500 lines, CI fails and the
code should be split before merging.

## Contributing And Releases

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local development workflow,
quality expectations, and public/private documentation rules.

See [CHANGELOG.md](CHANGELOG.md) for release notes. HydraSim uses semantic
versioning intent: patch releases for fixes, minor releases for compatible
features, and major releases for breaking public API or command changes.

Release tags matching `v*` build and validate wheel/source distributions.
Publishing to PyPI is available through the release workflow only when manually
dispatched on a release tag with the `publish-pypi` input enabled and the `pypi`
trusted-publishing environment configured.

## License

MIT. See [LICENSE](LICENSE).

## Security

HydraSim is simulation-only infrastructure. Do not use it to test systems you do
not own or administer. Report vulnerabilities and sensitive safety concerns
through the private process in [SECURITY.md](SECURITY.md).
