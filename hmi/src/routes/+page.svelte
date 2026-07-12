<script lang="ts">
  import { onMount } from 'svelte';
  import AlarmStrip from '$lib/components/AlarmStrip.svelte';
  import ControlPanel from '$lib/components/ControlPanel.svelte';
  import KpiTile from '$lib/components/KpiTile.svelte';
  import ProcessAreaGrid from '$lib/components/ProcessAreaGrid.svelte';
  import ScenarioPanel from '$lib/components/ScenarioPanel.svelte';
  import SystemStatus from '$lib/components/SystemStatus.svelte';
  import TogglePanel from '$lib/components/TogglePanel.svelte';
  import TrendChart from '$lib/components/TrendChart.svelte';
  import { fetchSnapshot, writeCoil, writeCommand } from '$lib/api';
  import { binaryDefaults, buildSnapshot, commandDefaults, scenarioCatalog } from '$lib/demoPlant';
  import type { BinaryValues, CommandValues, HmiSnapshot, ScenarioId } from '$lib/types';

  let elapsedSeconds = $state(0);
  let selectedScenario = $state<ScenarioId>('steady-state');
  let commandValues = $state<CommandValues>(commandDefaults());
  let binaryValues = $state<BinaryValues>(binaryDefaults());
  let liveSnapshot = $state<HmiSnapshot | null>(null);
  let lastError = $state<string | null>(null);

  const fallbackSnapshot = $derived(buildSnapshot(selectedScenario, elapsedSeconds, commandValues));
  const snapshot = $derived(
    liveSnapshot ?? { ...fallbackSnapshot, lastError: lastError ?? undefined },
  );
  const activeCommandValues = $derived(liveSnapshot?.commandValues ?? commandValues);
  const activeBinaryValues = $derived(liveSnapshot?.binaryValues ?? binaryValues);

  onMount(() => {
    void refreshSnapshot();

    const demoTimer = window.setInterval(() => {
      elapsedSeconds += 1;
    }, 1000);

    const runtimeTimer = window.setInterval(() => {
      void refreshSnapshot();
    }, 1000);

    return () => {
      window.clearInterval(demoTimer);
      window.clearInterval(runtimeTimer);
    };
  });

  async function refreshSnapshot(): Promise<void> {
    try {
      const next = await fetchSnapshot();
      liveSnapshot = next;
      commandValues = next.commandValues;
      binaryValues = next.binaryValues;
      lastError = null;
    } catch (error) {
      liveSnapshot = null;
      lastError = error instanceof Error ? error.message : 'Rust runtime unavailable';
    }
  }

  function selectScenario(scenario: ScenarioId): void {
    selectedScenario = scenario;
    elapsedSeconds = 0;
  }

  async function updateCommand(id: string, value: number): Promise<void> {
    commandValues = { ...activeCommandValues, [id]: value };
    if (liveSnapshot) {
      try {
        await writeCommand(id, value);
        await refreshSnapshot();
      } catch (error) {
        lastError = error instanceof Error ? error.message : 'command write failed';
      }
    }
  }

  async function updateCoil(id: string, enabled: boolean): Promise<void> {
    binaryValues = { ...activeBinaryValues, [id]: enabled };
    if (liveSnapshot) {
      try {
        await writeCoil(id, enabled);
        await refreshSnapshot();
      } catch (error) {
        lastError = error instanceof Error ? error.message : 'coil write failed';
      }
    }
  }

  function resetCommands(): void {
    commandValues = commandDefaults();
    binaryValues = binaryDefaults();
  }
</script>

<svelte:head>
  <title>HydraSim HMI</title>
  <meta
    name="description"
    content="HydraSim simulation-only human-machine interface for water-treatment process testing."
  />
</svelte:head>

<div class="app">
  <header class="topbar">
    <div class="brand">
      <span class="brand__mark" aria-hidden="true">HS</span>
      <div>
        <p>HydraSim HMI</p>
        <h1>Water Treatment Simulation Console</h1>
      </div>
    </div>
    <div class="topbar__meta" aria-label="Runtime status">
      <span class={`runtime-badge runtime-badge--${snapshot.connection}`}>
        {snapshot.connection}
      </span>
      <span class="simulation-label">Simulation Only · {snapshot.source}</span>
    </div>
  </header>

  <main class="dashboard">
    <SystemStatus {snapshot} />
    {#if snapshot.lastError}
      <p class="backend-error">Runtime fallback: {snapshot.lastError}</p>
    {/if}
    <AlarmStrip alarms={snapshot.alarms} />

    <section class="kpi-grid" aria-label="Process values">
      {#each snapshot.signals as signal (signal.tag)}
        <KpiTile {signal} />
      {/each}
    </section>

    <div class="workspace">
      <div class="main-stack">
        <ProcessAreaGrid areas={snapshot.areas} />
        <TrendChart points={snapshot.trends} />
      </div>

      <aside class="side-stack" aria-label="HMI controls">
        <ScenarioPanel
          scenarios={scenarioCatalog}
          selected={selectedScenario}
          onSelect={selectScenario}
        />
        <ControlPanel
          commands={snapshot.commands}
          values={activeCommandValues}
          onReset={resetCommands}
          onUpdate={updateCommand}
        />
        <TogglePanel
          commands={snapshot.binaryCommands}
          values={activeBinaryValues}
          onUpdate={updateCoil}
        />
      </aside>
    </div>
  </main>
</div>
