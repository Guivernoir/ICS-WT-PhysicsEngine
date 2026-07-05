import { describe, expect, it } from 'vitest';
import { clamp, scalePath, statusForSignal, worstStatus } from './processMath';
import type { ProcessSignal, TrendPoint } from './types';

const signal: ProcessSignal = {
  tag: 'AIT-CL-001',
  label: 'Chlorine residual',
  value: 1.2,
  unit: 'mg/L',
  decimals: 2,
  warningLow: 0.8,
  warningHigh: 2.4,
  alarmLow: 0.5,
  alarmHigh: 3.0,
  trend: 'steady',
};

describe('processMath', () => {
  it('clamps operator values inside configured bounds', () => {
    expect(clamp(12, 0, 10)).toBe(10);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(5, 0, 10)).toBe(5);
  });

  it('classifies process signals by warning and alarm limits', () => {
    expect(statusForSignal(signal)).toBe('normal');
    expect(statusForSignal({ ...signal, value: 0.7 })).toBe('warning');
    expect(statusForSignal({ ...signal, value: 3.4 })).toBe('alarm');
  });

  it('keeps alarm as the highest aggregate status', () => {
    expect(worstStatus(['normal', 'warning'])).toBe('warning');
    expect(worstStatus(['normal', 'alarm', 'warning'])).toBe('alarm');
    expect(worstStatus(['normal'])).toBe('normal');
  });

  it('builds an svg path for trend data', () => {
    const points: TrendPoint[] = [
      { minute: -2, ph: 7.1, chlorine: 1.0, flow: 4.0, turbidity: 0.2 },
      { minute: -1, ph: 7.2, chlorine: 1.1, flow: 4.2, turbidity: 0.3 },
    ];

    expect(
      scalePath(points, (point) => point.chlorine, { width: 100, height: 40, padding: 4 }),
    ).toMatchInlineSnapshot(`"M 4.0 36.0 L 96.0 4.0"`);
  });
});
