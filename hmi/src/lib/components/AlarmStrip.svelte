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
    {#each activeAlarms as alarm (alarm.id)}
      <article class={`alarm alarm--${alarm.severity}`}>
        <strong>{alarm.area}</strong>
        <span>{alarm.message}</span>
      </article>
    {/each}
  </div>
</section>

<style>
  .alarms {
    display: grid;
    gap: 0.75rem;
    padding: 0.85rem 1rem;
    border: 1px solid var(--border);
    background: var(--surface);
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
  }

  .alarms__list {
    display: grid;
    gap: 0.5rem;
  }

  .alarm {
    justify-content: flex-start;
    padding: 0.65rem 0.75rem;
    border-left: 0.35rem solid var(--ok);
    background: var(--surface-strong);
  }

  .alarm--warning {
    border-left-color: var(--warn);
  }

  .alarm--critical {
    border-left-color: var(--danger);
  }

  .alarm span {
    color: var(--text);
  }
</style>
