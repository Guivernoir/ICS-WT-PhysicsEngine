<script lang="ts">
  import { onMount } from 'svelte';
  import AlarmStrip from '$lib/components/AlarmStrip.svelte';
  import ControlPanel from '$lib/components/ControlPanel.svelte';
  import KpiTile from '$lib/components/KpiTile.svelte';
  import ProcessAreaGrid from '$lib/components/ProcessAreaGrid.svelte';
  import ScenarioPanel from '$lib/components/ScenarioPanel.svelte';
  import SystemStatus from '$lib/components/SystemStatus.svelte';
  import TrendChart from '$lib/components/TrendChart.svelte';
  import { buildSnapshot, commandDefaults, scenarioCatalog } from '$lib/demoPlant';
  import type { CommandValues, ScenarioId } from '$lib/types';

  let elapsedSeconds = $state(0);
  let selectedScenario = $state<ScenarioId>('steady-state');
  let commandValues = $state<CommandValues>(commandDefaults());
  const snapshot = $derived(buildSnapshot(selectedScenario, elapsedSeconds, commandValues));

  onMount(() => {
    const timer = window.setInterval(() => {
      elapsedSeconds += 1;
    }, 1000);

    return () => window.clearInterval(timer);
  });

  function selectScenario(scenario: ScenarioId): void {
    selectedScenario = scenario;
    elapsedSeconds = 0;
  }

  function updateCommand(id: string, value: number): void {
    commandValues = { ...commandValues, [id]: value };
  }

  function resetCommands(): void {
    commandValues = commandDefaults();
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
    <span class="simulation-label">Simulation Only · No Direct Modbus Control</span>
  </header>

  <main class="dashboard">
    <SystemStatus {snapshot} />
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
          values={commandValues}
          onReset={resetCommands}
          onUpdate={updateCommand}
        />
      </aside>
    </div>
  </main>
</div>
