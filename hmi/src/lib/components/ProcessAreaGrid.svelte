<script lang="ts">
  import { clamp, formatValue } from '$lib/processMath';
  import type { ProcessArea } from '$lib/types';

  let { areas }: { areas: readonly ProcessArea[] } = $props();
</script>

<section class="areas" aria-label="Process areas">
  {#each areas as area (area.id)}
    <article class={`area area--${area.status}`}>
      <div class="area__heading">
        <h2>{area.name}</h2>
        <span>{area.controllerMode}</span>
      </div>

      <dl>
        <div>
          <dt>Flow</dt>
          <dd>{formatValue(area.flowRate, 2)} {area.flowUnit}</dd>
        </div>
        <div>
          <dt>Level</dt>
          <dd>{formatValue(area.tankLevelPercent, 0)}%</dd>
        </div>
        <div>
          <dt>Residual</dt>
          <dd>{formatValue(area.residualMgL, 2)} mg/L</dd>
        </div>
        <div>
          <dt>Turbidity</dt>
          <dd>{formatValue(area.turbidityNtu, 2)} NTU</dd>
        </div>
      </dl>

      <div class="area__level" style={`--level: ${clamp(area.tankLevelPercent, 0, 100)}%`}>
        <span></span>
      </div>
    </article>
  {/each}
</section>

<style>
  .areas {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
  }

  .area {
    position: relative;
    overflow: hidden;
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    box-shadow: var(--shadow);
  }

  .area::before {
    position: absolute;
    inset: 0 0 auto;
    height: 0.3rem;
    content: '';
    background: var(--ok);
  }

  .area--warning::before {
    background: var(--warn);
  }

  .area--alarm::before {
    background: var(--danger);
  }

  .area__heading {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    justify-content: space-between;
  }

  h2 {
    margin: 0;
    font-size: 1rem;
  }

  .area__heading span {
    padding: 0.2rem 0.45rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--surface-strong);
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
  }

  dl {
    display: grid;
    gap: 0.5rem;
    margin: 0.85rem 0 0;
  }

  dl div {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
  }

  dt {
    color: var(--muted);
    font-size: 0.9rem;
  }

  dd {
    margin: 0;
    color: var(--text-strong);
    font-weight: 700;
  }

  .area__level {
    height: 0.55rem;
    margin-top: 0.9rem;
    overflow: hidden;
    border-radius: 999px;
    background: var(--surface-strong);
  }

  .area__level span {
    display: block;
    width: var(--level);
    height: 100%;
    border-radius: inherit;
    background: var(--accent);
  }

  @media (max-width: 900px) {
    .areas {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 560px) {
    .areas {
      grid-template-columns: 1fr;
    }
  }
</style>
