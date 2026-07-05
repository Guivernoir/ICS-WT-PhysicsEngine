<script lang="ts">
  import type { ScenarioId, ScenarioOption } from '$lib/types';

  let {
    scenarios,
    selected,
    onSelect,
  }: {
    scenarios: readonly ScenarioOption[];
    selected: ScenarioId;
    onSelect: (scenario: ScenarioId) => void;
  } = $props();
</script>

<section class="scenarios" aria-label="Scenario selector">
  <div class="scenarios__heading">
    <p>Scenario</p>
    <h2>HydraSim Runtime Driver</h2>
  </div>

  <div class="scenarios__grid">
    {#each scenarios as scenario (scenario.id)}
      <button
        class:selected={scenario.id === selected}
        type="button"
        onclick={() => onSelect(scenario.id)}
      >
        <strong>{scenario.name}</strong>
        <span>{scenario.summary}</span>
      </button>
    {/each}
  </div>
</section>

<style>
  .scenarios {
    display: grid;
    gap: 0.85rem;
    padding: 1rem;
    border: 1px solid var(--border);
    background: var(--surface);
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
    margin-top: 0.15rem;
    font-size: 1rem;
  }

  .scenarios__grid {
    display: grid;
    gap: 0.6rem;
  }

  button {
    display: grid;
    gap: 0.3rem;
    padding: 0.75rem;
    border: 1px solid var(--border);
    background: var(--surface-strong);
    color: var(--text);
    font: inherit;
    text-align: left;
    cursor: pointer;
  }

  button:hover,
  button.selected {
    border-color: var(--accent);
    background: #edf7fb;
  }

  span {
    color: var(--muted);
    font-size: 0.86rem;
  }
</style>
