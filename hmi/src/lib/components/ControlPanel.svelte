<script lang="ts">
  import { formatValue } from '$lib/processMath';
  import type { CommandValues, OperatorCommand } from '$lib/types';

  let {
    commands,
    values,
    onReset,
    onUpdate,
  }: {
    commands: readonly OperatorCommand[];
    values: CommandValues;
    onReset: () => void;
    onUpdate: (id: string, value: number) => void | Promise<void>;
  } = $props();

  function valueFor(command: OperatorCommand): number {
    return values[command.id] ?? command.min;
  }
</script>

<section class="controls" aria-label="Operator controls">
  <div class="controls__heading">
    <div>
      <p>Operator Setpoints</p>
      <h2>Bounded Simulation Commands</h2>
    </div>
    <button type="button" onclick={onReset}>Reset</button>
  </div>

  <div class="controls__list">
    {#each commands as command (command.id)}
      <label class="control">
        <span>
          <strong>{command.label}</strong>
          <small>{command.target} · {command.valueLabel}</small>
        </span>
        <input
          type="range"
          min={command.min}
          max={command.max}
          step={command.step}
          value={valueFor(command)}
          oninput={(event) => onUpdate(command.id, event.currentTarget.valueAsNumber)}
        />
        <output>{formatValue(valueFor(command), command.step < 1 ? 1 : 0)} {command.unit}</output>
      </label>
    {/each}
  </div>
</section>

<style>
  .controls {
    display: grid;
    gap: 1rem;
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    box-shadow: var(--shadow);
  }

  .controls__heading {
    display: flex;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
  }

  p,
  h2 {
    margin: 0;
  }

  p {
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  h2 {
    font-size: 1rem;
    color: var(--text-strong);
  }

  button {
    padding: 0.5rem 0.7rem;
    border: 1px solid var(--accent);
    border-radius: 6px;
    background: var(--surface-tinted);
    color: var(--accent-strong);
    font: inherit;
    font-weight: 800;
    cursor: pointer;
  }

  button:hover {
    background: #dff0ee;
  }

  .controls__list,
  .control {
    display: grid;
    gap: 0.8rem;
  }

  .control {
    grid-template-columns: minmax(10rem, 1fr) minmax(10rem, 1.4fr) 5rem;
    align-items: center;
    padding: 0.75rem 0;
    border-top: 1px solid var(--border);
  }

  .control span {
    display: grid;
    gap: 0.2rem;
  }

  small {
    color: var(--muted);
  }

  input[type='range'] {
    width: 100%;
    height: 0.45rem;
    border-radius: 999px;
    appearance: none;
    accent-color: var(--accent);
    background: linear-gradient(90deg, var(--accent), var(--surface-strong));
  }

  input[type='range']::-webkit-slider-thumb {
    width: 1.1rem;
    height: 1.1rem;
    border: 2px solid #ffffff;
    border-radius: 999px;
    appearance: none;
    background: var(--accent-strong);
    box-shadow: 0 2px 8px rgba(8, 127, 140, 0.28);
  }

  input[type='range']::-moz-range-thumb {
    width: 1.1rem;
    height: 1.1rem;
    border: 2px solid #ffffff;
    border-radius: 999px;
    background: var(--accent-strong);
    box-shadow: 0 2px 8px rgba(8, 127, 140, 0.28);
  }

  output {
    justify-self: end;
    padding: 0.22rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface-strong);
    color: var(--text-strong);
    font-weight: 700;
  }

  @media (max-width: 720px) {
    .control {
      grid-template-columns: 1fr;
    }

    output {
      justify-self: start;
    }
  }
</style>
