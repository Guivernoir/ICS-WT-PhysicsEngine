# HydraSim HMI

The HydraSim HMI is a SvelteKit operator console for simulation-only process
visibility and bounded setpoint experiments. It is intentionally separated from
the Python runtime: the browser UI does not speak Modbus and does not claim
real-plant control authority.

## Local Development

```bash
npm ci
npm run dev
```

The development server binds to `127.0.0.1` by default. The current dashboard
uses deterministic simulated telemetry while the Python HMI API boundary is
designed.

## Quality Gate

```bash
npm run lint
npm run check
npm run test
npm run build
```

CI runs the same commands. The root HydraSim quality checker also enforces the
500-line code-file limit for Svelte, TypeScript, JavaScript, CSS, Python, and
type-stub files.
