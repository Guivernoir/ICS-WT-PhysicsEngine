<script lang="ts">
  import { formatValue, statusForSignal } from '$lib/processMath';
  import type { ProcessSignal } from '$lib/types';

  let { signal }: { signal: ProcessSignal } = $props();
  const status = $derived(statusForSignal(signal));
  const statusLabel = $derived(status === 'alarm' ? 'Alarm' : status);
</script>

<article class={`kpi kpi--${status}`} aria-label={`${signal.label} ${status}`}>
  <div class="kpi__header">
    <div>
      <p class="kpi__tag">{signal.tag}</p>
      <h2>{signal.label}</h2>
    </div>
    <span>{statusLabel}</span>
  </div>
  <div class="kpi__reading">
    <span>{formatValue(signal.value, signal.decimals)}</span>
    <small>{signal.unit}</small>
  </div>
  <p class="kpi__trend">Trend: {signal.trend}</p>
</article>

<style>
  .kpi {
    position: relative;
    display: grid;
    gap: 0.75rem;
    min-height: 10rem;
    overflow: hidden;
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    box-shadow: var(--shadow);
  }

  .kpi::before {
    position: absolute;
    inset: 0 0 auto;
    height: 0.25rem;
    content: '';
    background: var(--ok);
  }

  .kpi--normal::before {
    background: var(--ok);
  }

  .kpi--warning::before {
    background: var(--warn);
  }

  .kpi--alarm::before {
    background: var(--danger);
  }

  .kpi__header {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    justify-content: space-between;
  }

  .kpi__header span {
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: var(--ok-bg);
    color: var(--ok-text);
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: capitalize;
  }

  .kpi--warning .kpi__header span {
    background: var(--warn-bg);
    color: var(--warn-text);
  }

  .kpi--alarm .kpi__header span {
    background: var(--danger-bg);
    color: var(--danger);
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
    line-height: 1.2;
  }

  .kpi__reading {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
  }

  .kpi__reading span {
    color: var(--text-strong);
    font-size: 2.15rem;
    font-weight: 700;
  }

  .kpi__reading small {
    color: var(--muted);
  }
</style>
