# HydraSim HMI

The HydraSim HMI is a SvelteKit operator console for simulation-only process
visibility and bounded setpoint experiments. It is intentionally UI-only: the
browser talks to the Rust runtime over HTTP and does not implement process,
PCS, or protocol behavior.

The dashboard includes status tiles, alarm banners, bounded setpoint controls,
coil switches, area summaries, and individual trend charts for pH, residual,
flow, and turbidity. Each chart has its own scale and latest-value indicator.

## Local Development

```bash
npm ci
npm run dev
```

The development server binds to `127.0.0.1` by default. To use live HydraSim
runtime values, run the Rust runtime in another shell:

```bash
cd ../runtime
HS_RUNTIME_PYTHON=../.venv/bin/python cargo run
```

In development mode, the frontend reads `VITE_HS_HMI_API_URL`, defaulting to
`http://127.0.0.1:8088`. In production builds served by the Rust runtime, API
calls use the same origin as the HMI.

To access the HMI directly through the Rust runtime:

```bash
npm run build
cd ../runtime
HS_RUNTIME_PYTHON=../.venv/bin/python cargo run
```

Open `http://127.0.0.1:8088`. When the runtime API is unavailable, the
dashboard falls back to deterministic demo telemetry.

To test the optional Modbus path alongside the HMI:

```bash
cd ../runtime
HS_RUNTIME_PYTHON=../.venv/bin/python HS_RUNTIME_MODBUS_ENABLED=true cargo run
```

## Quality Gate

```bash
npm run lint
npm run check
npm run test
npm run build
```

CI runs the same frontend commands. The root HydraSim quality checker also
enforces the 500-line code-file limit for Rust, Svelte, TypeScript, JavaScript,
CSS, Python, and type-stub files.
