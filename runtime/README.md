# HydraSim Runtime

Rust runtime for HydraSim systems behavior. It owns the HMI static host, HMI
HTTP API, PCS validation/interlocks, command state, optional Modbus TCP server,
and the Python simulation-worker lifecycle.

Python remains the physical simulation engine. The runtime starts
`hydrasim.simulation_worker` over stdin/stdout and sends it final PCS outputs;
the worker returns sensed process state.

## Run

```bash
cd ../hmi
npm run build
cd ../runtime
HS_RUNTIME_PYTHON=../.venv/bin/python cargo run
```

Open `http://127.0.0.1:8088` for the integrated HMI. The Rust runtime serves
the static Svelte build and the `/api/*` endpoints from the same origin.

Defaults:

- HTTP bind address: `127.0.0.1:8088`
- HMI static build directory: project `hmi/build`
- Modbus TCP: disabled
- Modbus bind address when enabled: `127.0.0.1:5502`
- Python executable: `python3`
- Python worker module: `hydrasim.simulation_worker`
- Python source path: project `src`
- Runtime tick: `1000` ms

Environment overrides:

```bash
HS_RUNTIME_BIND_ADDR=127.0.0.1:8088
HS_RUNTIME_HMI_DIR=../hmi/build
HS_RUNTIME_MODBUS_ENABLED=false
HS_RUNTIME_MODBUS_BIND_ADDR=127.0.0.1:5502
HS_RUNTIME_PYTHON=../.venv/bin/python
HS_RUNTIME_SIM_MODULE=hydrasim.simulation_worker
HS_RUNTIME_PYTHONPATH=../src
HS_RUNTIME_TICK_MS=1000
```

Enable Modbus only for PLC or Modbus-client integration work:

```bash
HS_RUNTIME_PYTHON=../.venv/bin/python HS_RUNTIME_MODBUS_ENABLED=true cargo run
```

## API

- `GET /api/health`
- `GET /api/snapshot`
- `PUT /api/commands/{id}` with `{"value": 0.2}`
- `PUT /api/coils/{id}` with `{"enabled": true}`

Smoke checks from another shell:

```bash
curl http://127.0.0.1:8088/api/health
curl http://127.0.0.1:8088/api/snapshot
curl -X PUT -H 'content-type: application/json' \
  -d '{"value": 0.25}' \
  http://127.0.0.1:8088/api/commands/chlorine-flow
```

With `HS_RUNTIME_MODBUS_ENABLED=true`, Modbus TCP listens on
`127.0.0.1:5502`. The HMI still talks to Rust over HTTP; browser code never
speaks Modbus directly.

Supported numeric command IDs:

- `acid-flow`
- `chlorine-flow`
- `inlet-flow`
- `acid-concentration`
- `chlorine-concentration`

Supported coil command IDs:

- `acid-pump-enable`
- `chlorine-pump-enable`
- `simulation-running`

## Quality

```bash
cargo fmt --check
cargo check --locked
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
```
