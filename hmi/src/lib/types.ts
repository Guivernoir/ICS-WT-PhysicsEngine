export type AlarmSeverity = 'notice' | 'warning' | 'critical';
export type ConnectionState = 'connected' | 'degraded' | 'offline';
export type ScenarioId = 'steady-state' | 'chlorine-upset' | 'filter-backwash' | 'loss-of-feed';
export type SignalStatus = 'normal' | 'warning' | 'alarm';

export interface ProcessSignal {
  tag: string;
  label: string;
  value: number;
  unit: string;
  decimals: number;
  warningLow?: number;
  warningHigh?: number;
  alarmLow?: number;
  alarmHigh?: number;
  trend: 'rising' | 'falling' | 'steady';
}

export interface ProcessArea {
  id: string;
  name: string;
  status: SignalStatus;
  flowMgd: number;
  tankLevelPercent: number;
  residualMgL: number;
  turbidityNtu: number;
  controllerMode: 'auto' | 'manual' | 'hold';
}

export interface AlarmEvent {
  id: string;
  severity: AlarmSeverity;
  area: string;
  message: string;
  active: boolean;
}

export interface OperatorCommand {
  id: string;
  label: string;
  target: string;
  valueLabel: string;
  min: number;
  max: number;
  step: number;
  unit: string;
}

export type CommandValues = Record<string, number>;

export interface TrendPoint {
  minute: number;
  ph: number;
  chlorine: number;
  flow: number;
  turbidity: number;
}

export interface ScenarioOption {
  id: ScenarioId;
  name: string;
  summary: string;
}

export interface HmiSnapshot {
  connection: ConnectionState;
  elapsedSeconds: number;
  mode: string;
  scenario: ScenarioOption;
  signals: readonly ProcessSignal[];
  areas: readonly ProcessArea[];
  alarms: readonly AlarmEvent[];
  commands: readonly OperatorCommand[];
  trends: readonly TrendPoint[];
}
