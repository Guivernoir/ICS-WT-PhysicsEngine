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
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
  }

  .status div {
    padding: 0.85rem 1rem;
    border: 1px solid var(--border);
    background: var(--surface);
  }

  p {
    margin: 0 0 0.2rem;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  strong {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .connection {
    color: var(--ok-text);
  }

  .connection--degraded {
    color: var(--warn-text);
  }

  .connection--offline {
    color: var(--danger);
  }

  @media (max-width: 760px) {
    .status {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
