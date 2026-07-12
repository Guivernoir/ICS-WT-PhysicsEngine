<script lang="ts">
  import type { BinaryCommand, BinaryValues } from '$lib/types';

  let {
    commands,
    values,
    onUpdate,
  }: {
    commands: readonly BinaryCommand[];
    values: BinaryValues;
    onUpdate: (id: string, enabled: boolean) => void | Promise<void>;
  } = $props();

  function enabledFor(command: BinaryCommand): boolean {
    return values[command.id] ?? false;
  }
</script>

<section class="toggles" aria-label="Discrete controls">
  <div class="toggles__heading">
    <p>Discrete Controls</p>
    <h2>HydraSim Coils</h2>
  </div>

  <div class="toggles__list">
    {#each commands as command (command.id)}
      <label class="toggle">
        <span>
          <strong>{command.label}</strong>
          <small>{command.target} · {command.valueLabel}</small>
        </span>
        <input
          type="checkbox"
          checked={enabledFor(command)}
          onchange={(event) => onUpdate(command.id, event.currentTarget.checked)}
        />
      </label>
    {/each}
  </div>
</section>

<style>
  .toggles {
    display: grid;
    gap: 1rem;
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    box-shadow: var(--shadow);
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

  .toggles__list {
    display: grid;
    gap: 0.8rem;
  }

  .toggle {
    display: flex;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0;
    border-top: 1px solid var(--border);
  }

  .toggle span {
    display: grid;
    gap: 0.2rem;
  }

  small {
    color: var(--muted);
  }

  input {
    position: relative;
    flex: 0 0 auto;
    width: 2.7rem;
    height: 1.45rem;
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    appearance: none;
    accent-color: var(--accent);
    background: var(--surface-strong);
    cursor: pointer;
  }

  input::before {
    position: absolute;
    top: 0.18rem;
    left: 0.18rem;
    width: 0.95rem;
    height: 0.95rem;
    border-radius: 999px;
    content: '';
    background: var(--border-strong);
    transition:
      transform 120ms ease,
      background 120ms ease;
  }

  input:checked {
    border-color: var(--accent);
    background: var(--surface-tinted);
  }

  input:checked::before {
    transform: translateX(1.2rem);
    background: var(--accent);
  }
</style>
