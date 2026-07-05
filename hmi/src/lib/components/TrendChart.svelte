<script lang="ts">
  import { scalePath } from '$lib/processMath';
  import type { TrendPoint } from '$lib/types';

  let { points }: { points: readonly TrendPoint[] } = $props();
  const scale = { width: 760, height: 230, padding: 22 };
  const chlorinePath = $derived(scalePath(points, (point) => point.chlorine, scale));
  const flowPath = $derived(scalePath(points, (point) => point.flow, scale));
  const turbidityPath = $derived(scalePath(points, (point) => point.turbidity, scale));
</script>

<section class="trend" aria-label="Process trends">
  <div class="trend__heading">
    <div>
      <p>Trend Window</p>
      <h2>Last 28 Minutes</h2>
    </div>
    <div class="trend__legend">
      <span class="legend legend--chlorine">Chlorine</span>
      <span class="legend legend--flow">Flow</span>
      <span class="legend legend--turbidity">Turbidity</span>
    </div>
  </div>

  <svg viewBox={`0 0 ${scale.width} ${scale.height}`} role="img" aria-label="Synthetic trends">
    <line x1="22" y1="208" x2="738" y2="208" />
    <line x1="22" y1="22" x2="22" y2="208" />
    <path class="path path--chlorine" d={chlorinePath} />
    <path class="path path--flow" d={flowPath} />
    <path class="path path--turbidity" d={turbidityPath} />
  </svg>
</section>

<style>
  .trend {
    display: grid;
    gap: 0.75rem;
    padding: 1rem;
    border: 1px solid var(--border);
    background: var(--surface);
  }

  .trend__heading {
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

  .trend__legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
  }

  .legend::before {
    display: inline-block;
    width: 0.7rem;
    height: 0.7rem;
    margin-right: 0.35rem;
    content: '';
    background: var(--ok);
  }

  .legend--chlorine::before {
    background: var(--accent);
  }

  .legend--flow::before {
    background: var(--ok);
  }

  .legend--turbidity::before {
    background: var(--warn);
  }

  svg {
    width: 100%;
    min-height: 14rem;
    border: 1px solid var(--border);
    background: linear-gradient(180deg, #ffffff 0%, #f5f8fb 100%);
  }

  line {
    stroke: var(--border-strong);
    stroke-width: 1;
  }

  .path {
    fill: none;
    stroke-width: 4;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .path--chlorine {
    stroke: var(--accent);
  }

  .path--flow {
    stroke: var(--ok);
  }

  .path--turbidity {
    stroke: var(--warn);
  }
</style>
