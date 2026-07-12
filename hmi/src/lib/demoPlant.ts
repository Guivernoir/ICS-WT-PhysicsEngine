import { clamp, statusForSignal, worstStatus } from './processMath';
import type {
  AlarmEvent,
  BinaryCommand,
  BinaryValues,
  CommandValues,
  HmiSnapshot,
  OperatorCommand,
  ProcessArea,
  ProcessSignal,
  ScenarioId,
  ScenarioOption,
  SignalStatus,
  TrendPoint,
} from './types';

interface ScenarioProfile extends ScenarioOption {
  connection: HmiSnapshot['connection'];
  chlorineOffset: number;
  flowOffset: number;
  turbidityOffset: number;
  phOffset: number;
  alarmHint: string;
}

const SCENARIOS: readonly ScenarioProfile[] = [
  {
    id: 'steady-state',
    name: 'Steady State',
    summary: 'Nominal disinfection cell with stable feed, residual, and tank level.',
    connection: 'connected',
    chlorineOffset: 0,
    flowOffset: 0,
    turbidityOffset: 0,
    phOffset: 0,
    alarmHint: 'No active process alarms.',
  },
  {
    id: 'chlorine-upset',
    name: 'Chlorine Upset',
    summary: 'Residual drift caused by chemical feed mismatch.',
    connection: 'degraded',
    chlorineOffset: -0.55,
    flowOffset: 0.05,
    turbidityOffset: 0.03,
    phOffset: 0.08,
    alarmHint: 'Low chlorine residual approaching operator review limits.',
  },
  {
    id: 'filter-backwash',
    name: 'Filter Backwash',
    summary: 'Temporary flow diversion with higher turbidity at filter outlet.',
    connection: 'connected',
    chlorineOffset: -0.1,
    flowOffset: -0.45,
    turbidityOffset: 0.36,
    phOffset: -0.03,
    alarmHint: 'Filter turbidity elevated during synthetic backwash sequence.',
  },
  {
    id: 'loss-of-feed',
    name: 'Loss Of Feed',
    summary: 'Chemical feed interruption with residual decay and hold mode.',
    connection: 'degraded',
    chlorineOffset: -0.92,
    flowOffset: -0.2,
    turbidityOffset: 0.06,
    phOffset: 0.14,
    alarmHint: 'Chemical feed unavailable; simulated controller is in hold.',
  },
];

const COMMANDS: readonly OperatorCommand[] = [
  {
    id: 'acid-flow',
    label: 'Acid Flow',
    target: 'HR 40001',
    valueLabel: 'Setpoint',
    min: 0,
    max: 2,
    step: 0.01,
    unit: 'L/min',
  },
  {
    id: 'chlorine-flow',
    label: 'Chlorine Flow',
    target: 'HR 40003',
    valueLabel: 'Setpoint',
    min: 0,
    max: 1,
    step: 0.01,
    unit: 'L/min',
  },
  {
    id: 'inlet-flow',
    label: 'Inlet Flow',
    target: 'HR 40005',
    valueLabel: 'Flow setpoint',
    min: 0,
    max: 20,
    step: 0.1,
    unit: 'L/min',
  },
  {
    id: 'chlorine-concentration',
    label: 'Chlorine Stock',
    target: 'HR 40013',
    valueLabel: 'Concentration',
    min: 0,
    max: 200,
    step: 1,
    unit: 'mg/L',
  },
];

const BINARY_COMMANDS: readonly BinaryCommand[] = [
  {
    id: 'acid-pump-enable',
    label: 'Acid Pump',
    target: 'Coil 00001',
    valueLabel: 'Enable',
  },
  {
    id: 'chlorine-pump-enable',
    label: 'Chlorine Pump',
    target: 'Coil 00002',
    valueLabel: 'Enable',
  },
  {
    id: 'simulation-running',
    label: 'Simulation Running',
    target: 'Coil 00003',
    valueLabel: 'Run state',
  },
];

export const scenarioCatalog: readonly ScenarioOption[] = SCENARIOS.map(
  ({ id, name, summary }) => ({ id, name, summary }),
);

export function commandDefaults(): CommandValues {
  return {
    'acid-flow': 0.0,
    'chlorine-flow': 0.2,
    'inlet-flow': 5.0,
    'chlorine-concentration': 60.0,
  };
}

export function binaryDefaults(): BinaryValues {
  return {
    'acid-pump-enable': true,
    'chlorine-pump-enable': true,
    'simulation-running': true,
  };
}

export function buildSnapshot(
  scenarioId: ScenarioId,
  elapsedSeconds: number,
  commands: CommandValues,
): HmiSnapshot {
  const profile = SCENARIOS.find((item) => item.id === scenarioId) ?? SCENARIOS[0];
  const wave = Math.sin(elapsedSeconds / 9);
  const slowWave = Math.sin(elapsedSeconds / 24);
  const acidFlow = clamp(commands['acid-flow'] ?? 0, 0, 2);
  const chlorineFlow = clamp(commands['chlorine-flow'] ?? 0.2, 0, 1);
  const flowSetpoint = clamp(commands['inlet-flow'] ?? 5, 0, 20);

  const flow = flowSetpoint + profile.flowOffset + wave * 0.05;
  const chlorine = 1.38 + profile.chlorineOffset + chlorineFlow * 0.45 - flow * 0.035;
  const ph = 7.32 + profile.phOffset + slowWave * 0.03 - acidFlow * 0.08;
  const turbidity = 0.18 + profile.turbidityOffset + Math.max(flow - 8, 0) * 0.02;
  const tankLevel = clamp(72 + profile.flowOffset * 6 + slowWave * 2, 35, 96);

  const signals: readonly ProcessSignal[] = [
    signal('FIT-INT-001', 'Influent Flow', flow, 'L/min', 2, 1, 18, 0.1, 20),
    signal('AIT-PH-001', 'Outlet pH', ph, 'pH', 2, 6.8, 7.8, 6.5, 8.2),
    signal('AIT-CL-001', 'Outlet Chlorine', chlorine, 'mg/L', 2, 0.9, 2.2, 0.6, 2.8),
    signal('AIT-TU-001', 'Filter Turbidity', turbidity, 'NTU', 2, undefined, 0.5, undefined, 0.9),
    signal('LIT-CW-001', 'Clearwell Level', tankLevel, '%', 0, 45, 92, 35, 97),
  ];

  const status = worstStatus(signals.map(statusForSignal));
  const areas = buildAreas(status, flow, chlorine, turbidity, tankLevel, scenarioId);
  const alarms = buildAlarms(profile, signals, status);

  return {
    source: 'demo',
    connection: profile.connection,
    elapsedSeconds,
    mode: 'Simulation HMI',
    scenario: profile,
    signals,
    areas,
    alarms,
    commands: COMMANDS,
    commandValues: { ...commands },
    binaryCommands: BINARY_COMMANDS,
    binaryValues: binaryDefaults(),
    trends: buildTrends(profile, elapsedSeconds, flowSetpoint, chlorineFlow),
  };
}

function signal(
  tag: string,
  label: string,
  value: number,
  unit: string,
  decimals: number,
  warningLow?: number,
  warningHigh?: number,
  alarmLow?: number,
  alarmHigh?: number,
): ProcessSignal {
  const trend = value > 1 ? 'rising' : value < 0.8 ? 'falling' : 'steady';
  return { tag, label, value, unit, decimals, warningLow, warningHigh, alarmLow, alarmHigh, trend };
}

function buildAreas(
  status: SignalStatus,
  flow: number,
  chlorine: number,
  turbidity: number,
  tankLevel: number,
  scenarioId: ScenarioId,
): readonly ProcessArea[] {
  const holdMode = scenarioId === 'loss-of-feed';
  return [
    area('intake', 'Intake', 'normal', flow, 68, chlorine + 0.12, 0.22, 'auto'),
    area('filtration', 'Filtration', status, flow - 0.12, 64, chlorine + 0.02, turbidity, 'auto'),
    area(
      'disinfection',
      'Disinfection',
      status,
      flow - 0.18,
      tankLevel,
      chlorine,
      0.16,
      holdMode ? 'hold' : 'auto',
    ),
    area(
      'storage',
      'Storage & Pumping',
      'normal',
      flow - 0.2,
      tankLevel - 3,
      chlorine - 0.05,
      0.13,
      'auto',
    ),
  ];
}

function area(
  id: string,
  name: string,
  status: SignalStatus,
  flowRate: number,
  tankLevelPercent: number,
  residualMgL: number,
  turbidityNtu: number,
  controllerMode: ProcessArea['controllerMode'],
): ProcessArea {
  return {
    id,
    name,
    status,
    flowRate,
    flowUnit: 'L/min',
    tankLevelPercent,
    residualMgL,
    turbidityNtu,
    controllerMode,
  };
}

function buildAlarms(
  profile: ScenarioProfile,
  signals: readonly ProcessSignal[],
  status: SignalStatus,
): readonly AlarmEvent[] {
  const activeSignal = signals.find((item) => statusForSignal(item) !== 'normal');
  const severity = status === 'alarm' ? 'critical' : status === 'warning' ? 'warning' : 'notice';
  return [
    {
      id: 'simulation-boundary',
      severity: 'notice',
      area: 'HMI',
      message: 'HydraSim HMI is attached to simulated process data only.',
      active: true,
    },
    {
      id: `scenario-${profile.id}`,
      severity,
      area: activeSignal?.tag ?? 'Process',
      message: activeSignal ? `${activeSignal.label}: ${profile.alarmHint}` : profile.alarmHint,
      active: profile.id !== 'steady-state' || status !== 'normal',
    },
  ];
}

function buildTrends(
  profile: ScenarioProfile,
  elapsedSeconds: number,
  flowSetpoint: number,
  chlorineFlow: number,
): readonly TrendPoint[] {
  return Array.from({ length: 28 }, (_, index) => {
    const minute = index - 27;
    const phase = (elapsedSeconds + minute * 60) / 18;
    return {
      minute,
      ph: 7.32 + profile.phOffset + Math.sin(phase / 2) * 0.04,
      chlorine: 1.38 + profile.chlorineOffset + chlorineFlow * 0.3 + Math.sin(phase) * 0.08,
      flow: flowSetpoint + profile.flowOffset + Math.cos(phase / 1.8) * 0.08,
      turbidity: 0.18 + profile.turbidityOffset + Math.max(Math.sin(phase / 3), 0) * 0.06,
    };
  });
}
