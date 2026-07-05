import type { ProcessSignal, SignalStatus, TrendPoint } from './types';

export interface ChartScale {
  width: number;
  height: number;
  padding: number;
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
  if (points.length === 0) {
    return '';
  }

  const values = points.map(valueFor);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const xStep = points.length === 1 ? 0 : (scale.width - scale.padding * 2) / (points.length - 1);

  return points
    .map((point, index) => {
      const x = scale.padding + index * xStep;
      const normalized = (valueFor(point) - min) / range;
      const y = scale.height - scale.padding - normalized * (scale.height - scale.padding * 2);
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');
}
