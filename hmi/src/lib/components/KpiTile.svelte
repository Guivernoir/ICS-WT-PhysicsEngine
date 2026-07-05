<script lang="ts">
  import { formatValue, statusForSignal } from '$lib/processMath';
  import type { ProcessSignal } from '$lib/types';

  let { signal }: { signal: ProcessSignal } = $props();
  const status = $derived(statusForSignal(signal));
</script>

<article class={`kpi kpi--${status}`} aria-label={`${signal.label} ${status}`}>
  <div>
    <p class="kpi__tag">{signal.tag}</p>
    <h2>{signal.label}</h2>
  </div>
  <div class="kpi__reading">
    <span>{formatValue(signal.value, signal.decimals)}</span>
    <small>{signal.unit}</small>
  </div>
  <p class="kpi__trend">Trend: {signal.trend}</p>
</article>

<style>
  .kpi {
    display: grid;
    gap: 0.75rem;
    min-height: 10rem;
    padding: 1rem;
    border: 1px solid var(--border);
    border-left-width: 0.4rem;
    background: var(--surface);
  }

  .kpi--normal {
    border-left-color: var(--ok);
  }

  .kpi--warning {
    border-left-color: var(--warn);
  }

  .kpi--alarm {
    border-left-color: var(--danger);
  }

  .kpi__tag,
  .kpi__trend {
    margin: 0;
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  h2 {
    margin: 0.2rem 0 0;
    font-size: 1rem;
  }

  .kpi__reading {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
  }

  .kpi__reading span {
    font-size: 2rem;
    font-weight: 700;
  }

  .kpi__reading small {
    color: var(--muted);
  }
</style>
