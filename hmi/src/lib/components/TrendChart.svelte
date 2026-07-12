<script lang="ts">
  import { formatValue, scaleAreaPath, scaleLatestPoint, scalePath } from '$lib/processMath';
  import type { ChartPoint } from '$lib/processMath';
  import type { TrendPoint } from '$lib/types';

  type TrendId = 'ph' | 'chlorine' | 'flow' | 'turbidity';

  interface TrendSeries {
    id: TrendId;
    label: string;
    unit: string;
    decimals: number;
    latest: string;
    range: string;
    path: string;
    areaPath: string;
    point: ChartPoint | null;
  }

  let { points }: { points: readonly TrendPoint[] } = $props();

  const chartScale = { width: 360, height: 130, padding: 18 };
  const series = $derived([
    buildSeries('ph', 'pH', 'pH', 2, (point) => point.ph),
    buildSeries('chlorine', 'Chlorine', 'mg/L', 2, (point) => point.chlorine),
    buildSeries('flow', 'Flow', 'L/min', 2, (point) => point.flow),
    buildSeries('turbidity', 'Turbidity', 'NTU', 2, (point) => point.turbidity),
  ]);

  function buildSeries(
    id: TrendId,
    label: string,
    unit: string,
    decimals: number,
    valueFor: (point: TrendPoint) => number,
  ): TrendSeries {
    const values = points.map(valueFor);
    const latest = values.at(-1);
    const minimum = values.length ? Math.min(...values) : null;
    const maximum = values.length ? Math.max(...values) : null;

    return {
      id,
      label,
      unit,
      decimals,
      latest: latest === undefined ? '--' : formatValue(latest, decimals),
      range:
        minimum === null || maximum === null
          ? '--'
          : `${formatValue(minimum, decimals)} - ${formatValue(maximum, decimals)}`,
      path: scalePath(points, valueFor, chartScale),
      areaPath: scaleAreaPath(points, valueFor, chartScale),
      point: scaleLatestPoint(points, valueFor, chartScale),
    };
  }
</script>

<section class="trend" aria-label="Process trends">
  <div class="trend__heading">
    <div>
      <p>Trend Window</p>
      <h2>Process Signal Trends</h2>
      <small>Last 28 minutes, each signal has its own chart and scale</small>
    </div>
    <div class="trend__legend">
      <span class="legend legend--ph">pH</span>
      <span class="legend legend--chlorine">Chlorine</span>
      <span class="legend legend--flow">Flow</span>
      <span class="legend legend--turbidity">Turbidity</span>
    </div>
  </div>

  <div class="trend__charts" aria-label="Individual process charts">
    {#each series as signal (signal.id)}
      <article class={`chart chart--${signal.id}`}>
        <div class="chart__header">
          <div>
            <h3>{signal.label}</h3>
            <span>{signal.range} {signal.unit}</span>
          </div>
          <div class={`indicator indicator--${signal.id}`}>
            <strong>{signal.latest}</strong>
            <small>{signal.unit}</small>
          </div>
        </div>

        <svg
          viewBox={`0 0 ${chartScale.width} ${chartScale.height}`}
          role="img"
          aria-label={`${signal.label} trend`}
        >
          <rect class="chart__surface" x="0.5" y="0.5" width="359" height="129" rx="7" />
          <line
            class="chart__grid"
            x1={chartScale.padding}
            y1={chartScale.height / 2}
            x2={chartScale.width - chartScale.padding}
            y2={chartScale.height / 2}
          />
          <line
            class="chart__axis"
            x1={chartScale.padding}
            y1={chartScale.height - chartScale.padding}
            x2={chartScale.width - chartScale.padding}
            y2={chartScale.height - chartScale.padding}
          />
          <path class={`area area--${signal.id}`} d={signal.areaPath} />
          <path class={`path path--${signal.id}`} d={signal.path} />
          {#if signal.point}
            <circle
              class={`marker marker--${signal.id}`}
              cx={signal.point.x}
              cy={signal.point.y}
              r="4.5"
            />
          {/if}
        </svg>
      </article>
    {/each}
  </div>
</section>

<style>
  .trend {
    display: grid;
    gap: 0.8rem;
    padding: 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    box-shadow: var(--shadow);
  }

  .trend__heading {
    display: flex;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
  }

  p,
  h2,
  small {
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

  .trend__heading small {
    display: block;
    margin-top: 0.2rem;
    color: var(--muted);
    font-size: 0.82rem;
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
    border-radius: 999px;
    content: '';
    background: var(--ok);
  }

  .legend--ph::before {
    background: #4c6f63;
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

  .trend__charts {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
  }

  .chart {
    display: grid;
    gap: 0.65rem;
    padding: 0.75rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--surface-strong);
  }

  .chart__header {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    justify-content: space-between;
  }

  h3 {
    margin: 0;
    color: var(--text-strong);
    font-size: 0.95rem;
  }

  .chart__header span {
    color: var(--muted);
    font-size: 0.78rem;
  }

  svg {
    width: 100%;
    min-height: 8.4rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: #ffffff;
  }

  .chart__surface {
    fill: #ffffff;
    stroke: var(--border);
  }

  .chart__grid {
    stroke: var(--border);
    stroke-dasharray: 4 6;
  }

  .chart__axis {
    stroke: var(--border-strong);
  }

  .area {
    opacity: 0.18;
  }

  .area--ph {
    fill: #4c6f63;
  }

  .area--chlorine {
    fill: var(--accent);
  }

  .area--flow {
    fill: var(--ok);
  }

  .area--turbidity {
    fill: var(--warn);
  }

  .path {
    fill: none;
    stroke-width: 3.3;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .path--ph {
    stroke: #4c6f63;
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

  .marker {
    stroke: #ffffff;
    stroke-width: 2;
    fill: #4c6f63;
  }

  .marker--chlorine {
    fill: var(--accent);
  }

  .marker--flow {
    fill: var(--ok);
  }

  .marker--turbidity {
    fill: var(--warn);
  }

  .indicator {
    position: relative;
    display: grid;
    gap: 0.08rem;
    min-width: 5.9rem;
    padding: 0.45rem 0.65rem 0.45rem 0.85rem;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: #ffffff;
  }

  .indicator::before {
    position: absolute;
    inset: 0.7rem auto 0.7rem 0;
    width: 0.28rem;
    border-radius: 999px;
    content: '';
    background: #4c6f63;
  }

  .indicator--chlorine::before {
    background: var(--accent);
  }

  .indicator--flow::before {
    background: var(--ok);
  }

  .indicator--turbidity::before {
    background: var(--warn);
  }

  .indicator small {
    color: var(--muted);
    font-size: 0.76rem;
    font-style: normal;
  }

  .indicator strong {
    color: var(--text-strong);
    font-size: 1.12rem;
  }

  @media (max-width: 900px) {
    .trend__charts {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 760px) {
    .trend__heading {
      align-items: flex-start;
      flex-direction: column;
    }
  }
</style>
