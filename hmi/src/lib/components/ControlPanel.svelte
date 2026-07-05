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
    onUpdate: (id: string, value: number) => void;
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
    background: var(--surface);
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
  }

  button {
    padding: 0.5rem 0.7rem;
    border: 1px solid var(--border-strong);
    background: var(--surface-strong);
    color: var(--text);
    font: inherit;
    cursor: pointer;
  }

  button:hover {
    border-color: var(--accent);
  }

  .controls__list,
  .control {
    display: grid;
    gap: 0.8rem;
  }

  .control {
    grid-template-columns: minmax(10rem, 1fr) minmax(10rem, 1.4fr) 5rem;
    align-items: center;
    padding-top: 0.85rem;
    border-top: 1px solid var(--border);
  }

  .control span {
    display: grid;
    gap: 0.2rem;
  }

  small {
    color: var(--muted);
  }

  input {
    width: 100%;
    accent-color: var(--accent);
  }

  output {
    justify-self: end;
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
