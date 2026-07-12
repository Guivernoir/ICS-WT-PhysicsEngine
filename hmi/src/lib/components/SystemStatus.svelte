<script lang="ts">
  import type { HmiSnapshot } from '$lib/types';

  let { snapshot }: { snapshot: HmiSnapshot } = $props();
  const elapsedMinutes = $derived(Math.floor(snapshot.elapsedSeconds / 60));
  const elapsedSeconds = $derived(snapshot.elapsedSeconds % 60);
</script>

<section class="status" aria-label="System status">
  <div>
    <p>Mode</p>
    <strong>{snapshot.mode}</strong>
  </div>
  <div>
    <p>Source</p>
    <strong>{snapshot.source}</strong>
  </div>
  <div>
    <p>Connection</p>
    <strong class={`connection connection--${snapshot.connection}`}>{snapshot.connection}</strong>
  </div>
  <div>
    <p>Scenario</p>
    <strong>{snapshot.scenario.name}</strong>
  </div>
  <div>
    <p>Runtime</p>
    <strong>{elapsedMinutes}m {elapsedSeconds}s</strong>
  </div>
</section>

<style>
  .status {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.75rem;
  }

  .status div {
    position: relative;
    overflow: hidden;
    min-height: 5.2rem;
    padding: 0.9rem 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    box-shadow: var(--shadow);
  }

  .status div::before {
    position: absolute;
    inset: 0 auto 0 0;
    width: 0.25rem;
    content: '';
    background: var(--accent);
  }

  .status div:nth-child(3)::before {
    background: var(--ok);
  }

  p {
    margin: 0 0 0.35rem;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  strong {
    display: block;
    overflow: hidden;
    color: var(--text-strong);
    font-size: 1.02rem;
    line-height: 1.25;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .connection {
    display: inline-flex;
    width: fit-content;
    align-items: center;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    background: var(--ok-bg);
    color: var(--ok-text);
    text-transform: capitalize;
  }

  .connection--degraded {
    background: var(--warn-bg);
    color: var(--warn-text);
  }

  .connection--offline {
    background: var(--danger-bg);
    color: var(--danger);
  }

  @media (max-width: 760px) {
    .status {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
