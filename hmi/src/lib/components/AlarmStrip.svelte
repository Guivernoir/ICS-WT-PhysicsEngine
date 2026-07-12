<script lang="ts">
  import type { AlarmEvent } from '$lib/types';

  let { alarms }: { alarms: readonly AlarmEvent[] } = $props();
  const activeAlarms = $derived(alarms.filter((alarm) => alarm.active));
</script>

<section class="alarms" aria-label="Active alarms">
  <div class="alarms__header">
    <strong>Alarm Banner</strong>
    <span>{activeAlarms.length} active</span>
  </div>

  <div class="alarms__list">
    {#if activeAlarms.length > 0}
      {#each activeAlarms as alarm (alarm.id)}
        <article class={`alarm alarm--${alarm.severity}`}>
          <strong>{alarm.area}</strong>
          <span>{alarm.message}</span>
        </article>
      {/each}
    {:else}
      <article class="alarm alarm--clear">
        <strong>Clear</strong>
        <span>No active simulation alarms.</span>
      </article>
    {/if}
  </div>
</section>

<style>
  .alarms {
    display: grid;
    gap: 0.75rem;
    padding: 0.9rem 1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    box-shadow: var(--shadow);
  }

  .alarms__header,
  .alarm {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    justify-content: space-between;
  }

  .alarms__header span {
    color: var(--muted);
    font-size: 0.85rem;
    font-weight: 700;
  }

  .alarms__list {
    display: grid;
    gap: 0.5rem;
  }

  .alarm {
    justify-content: flex-start;
    padding: 0.68rem 0.8rem;
    border: 1px solid var(--border);
    border-left: 0.35rem solid var(--ok);
    border-radius: 7px;
    background: var(--ok-bg);
  }

  .alarm--clear,
  .alarm--notice {
    border-left-color: var(--ok);
    background: var(--ok-bg);
  }

  .alarm--warning {
    border-left-color: var(--warn);
    background: var(--warn-bg);
  }

  .alarm--critical {
    border-left-color: var(--danger);
    background: var(--danger-bg);
  }

  .alarm strong {
    min-width: 5rem;
    color: var(--text-strong);
  }

  .alarm span {
    color: var(--text);
  }
</style>
