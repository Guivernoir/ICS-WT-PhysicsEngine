import type { ProcessSignal, SignalStatus, TrendPoint } from './types';

export interface ChartScale {
  width: number;
  height: number;
  padding: number;
}

export interface ChartPoint {
  x: number;
  y: number;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function statusForSignal(signal: ProcessSignal): SignalStatus {
  if (
    (signal.alarmLow !== undefined && signal.value < signal.alarmLow) ||
    (signal.alarmHigh !== undefined && signal.value > signal.alarmHigh)
  ) {
    return 'alarm';
  }

  if (
    (signal.warningLow !== undefined && signal.value < signal.warningLow) ||
    (signal.warningHigh !== undefined && signal.value > signal.warningHigh)
  ) {
    return 'warning';
  }

  return 'normal';
}

export function worstStatus(statuses: readonly SignalStatus[]): SignalStatus {
  if (statuses.includes('alarm')) {
    return 'alarm';
  }
  if (statuses.includes('warning')) {
    return 'warning';
  }
  return 'normal';
}

export function formatValue(value: number, decimals: number): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function scalePath(
  points: readonly TrendPoint[],
  valueFor: (point: TrendPoint) => number,
  scale: ChartScale,
): string {
  return scalePoints(points, valueFor, scale)
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`)
    .join(' ');
}

export function scaleAreaPath(
  points: readonly TrendPoint[],
  valueFor: (point: TrendPoint) => number,
  scale: ChartScale,
): string {
  if (points.length === 0) {
    return '';
  }

  const scaled = scalePoints(points, valueFor, scale);
  const first = scaled[0];
  const last = scaled[scaled.length - 1];
  const baseline = scale.height - scale.padding;
  const line = scaled
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`)
    .join(' ');

  return `${line} L ${last.x.toFixed(1)} ${baseline.toFixed(1)} L ${first.x.toFixed(1)} ${baseline.toFixed(1)} Z`;
}

export function scaleLatestPoint(
  points: readonly TrendPoint[],
  valueFor: (point: TrendPoint) => number,
  scale: ChartScale,
): ChartPoint | null {
  const scaled = scalePoints(points, valueFor, scale);
  return scaled.at(-1) ?? null;
}

function scalePoints(
  points: readonly TrendPoint[],
  valueFor: (point: TrendPoint) => number,
  scale: ChartScale,
): ChartPoint[] {
  if (points.length === 0) {
    return [];
  }

  const values = points.map(valueFor);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min;
  const xStep = points.length === 1 ? 0 : (scale.width - scale.padding * 2) / (points.length - 1);

  return points.map((point, index) => {
    const x = scale.padding + index * xStep;
    const normalized = range === 0 ? 0.5 : (valueFor(point) - min) / range;
    const y = scale.height - scale.padding - normalized * (scale.height - scale.padding * 2);
    return { x, y };
  });
}
