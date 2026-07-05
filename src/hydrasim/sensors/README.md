# Sensors Module - Water Treatment Reactor

## Overview

This module provides sensor simulations for the water treatment reactor physics engine. It models measurement characteristics including noise, response lag, calibration drift, and basic sensor-specific effects.

## Architecture

```
sensors/
├── base_sensor.py          # Abstract base class with common functionality
├── ph_sensor.py            # Glass electrode pH measurement
├── chlorine_sensor.py      # Amperometric and DPD colorimetric chlorine
├── flow_sensor.py          # Turbine and magnetic flowmeters
├── temperature_sensor.py   # RTD and thermocouple temperature
└── __init__.py             # Package initialization and convenience functions
```

## Key Features

- Gaussian noise based on sensor precision
- First-order response dynamics (lag)
- Calibration drift over time
- Warm-up periods after power-on or calibration
- Hysteresis (direction-dependent behavior)
- Sample line transport delays
- Installation effects (flow velocity, air bubbles, grounding quality)
- Fault states (out-of-range, saturation, etc.)
- Calibration history and expiration

## Sensor Specifications

### pH Sensor (Glass Electrode)

```python
from sensors import pHSensor

sensor = pHSensor(
    name="pH_inlet",
    zone_index=0,
    precision=0.01,      # ±0.01 pH (1σ noise)
    response_time=10.0,  # 10 second time constant
    drift_rate=0.01/24   # 0.01 pH per day
)
```

**Specifications:**
- Range: 0-14 pH
- Precision: ±0.01 pH (1σ)
- Accuracy: ±0.05 pH after calibration
- Response Time: 5-30 seconds (t90)
- Drift: ±0.01 pH/day typical

**Effects Modeled:**
- Temperature compensation (Nernst equation)
- Electrical noise
- Calibration offset and slope

### Chlorine Sensor

Supports two types:

**Amperometric**
```python
from sensors import ChlorineSensor, ChlorineSensorType

sensor = ChlorineSensor(
    name="Cl_inlet",
    zone_index=0,
    sensor_type=ChlorineSensorType.AMPEROMETRIC
)
```

**DPD Colorimetric**
```python
sensor = ChlorineSensor(
    name="Cl_outlet",
    zone_index=-1,
    sensor_type=ChlorineSensorType.DPD_COLORIMETRIC
)
```

**Specifications:**
- Range: 0-10 mg/L
- Precision: ±0.01 mg/L (amperometric), ±0.02 mg/L (DPD)
- Accuracy: ±0.05 mg/L
- Response Time: 30 s (amperometric), 60-120 s (DPD)
- Drift: ±0.02 mg/L/day (amperometric), ±0.01 mg/L/day (DPD)

**Effects Modeled:**
- pH-dependent speciation (HOCl vs OCl⁻)
- Temperature effects
- Cross-sensitivities (amperometric only)

### Flow Sensor

**Turbine or Magnetic**
```python
from sensors import FlowSensor, FlowSensorType

sensor = FlowSensor(
    name="flow_main",
    sensor_type=FlowSensorType.MAGNETIC,
    full_scale=100.0
)
```

**Specifications:**
- Range: 0 to full_scale
- Precision: 1% FS (turbine), 0.5% FS (magnetic)
- Response Time: 0.5 s

**Effects Modeled:**
- Air bubble detection
- Conductivity threshold (magnetic)

### Temperature Sensor

**RTD or Thermocouple**
```python
from sensors import TemperatureSensor, TemperatureSensorType

sensor = TemperatureSensor(
    name="temp_inlet",
    zone_index=0,
    sensor_type=TemperatureSensorType.RTD_PT100
)
```

**Specifications:**
- Range: -10 to 110 °C
- Precision: ±0.1 °C (RTD), ±0.5 °C (thermocouple)

**Effects Modeled:**
- Self-heating (RTD)
- Cold junction compensation (thermocouple)

## Usage Examples

### Basic Reading

```python
reading = sensor.read(reactor.state)
print(f"Value: {reading.value:.3f} ± {reading.uncertainty:.3f}")
if reading.status != SensorStatus.NORMAL:
    print(f"Status: {reading.status.value}")
```

### Create Full Sensor Suite

```python
from sensors import create_realistic_sensor_suite

sensors = create_realistic_sensor_suite(reactor_config)
```

### Calibration Example

```python
# Single-point (used by MaintenanceManager for remote recal)
sensor.calibrate(
    reference_value=7.0,
    current_time=time.monotonic(),
    operator_id="modbus_remote",
    skip_warmup=True,   # sensor already warm — don't restart the warm-up timer
)

# Two-point pH (higher accuracy, also supports skip_warmup)
sensor.calibrate_two_point(
    ref_low=4.0, meas_low=4.05,
    ref_high=7.0, meas_high=7.02,
    timestamp=time.monotonic(),
    skip_warmup=True,
)
```

## Maintenance Integration

The sensors module is designed to be driven by `hydrasim.maintenance.MaintenanceManager`. Each sensor exposes a stable API that the manager calls over Modbus.

### Exposed maintenance API per sensor type

| Sensor type | Method | Called by action |
|---|---|---|
| All | `calibrate(ref, ts, skip_warmup=...)` | `CALIBRATE` |
| All | `reset()` | `FULL_RESET` |
| pH | `clean_electrode("water_rinse", ts)` | `CLEAN_WATER` |
| pH | `clean_electrode("acid_clean", ts)` | `CLEAN_ACID` |
| Chlorine (amperometric) | `replace_membrane(ts)` | `REPLACE_MEMBRANE` |
| Chlorine (DPD) | `replace_reagent(ts)` | `REPLACE_REAGENT` |

### Behaviour after hardware replacement

`replace_membrane()` and `replace_reagent()` do **not** calibrate the sensor to a known value. Instead they:
- Reset wear counters and potency to factory state
- Clear `calibration_offset` to zero (no artificial bias)
- Append a `CalibrationRecord` with `validity_hours=0` so `_check_calibration_valid()` returns `False` immediately
- Set `status = CALIBRATION_EXPIRED`

The sensor will continue to produce readings (using the cleared offset) but the CALIBRATION_EXPIRED status will be visible to SCADA/HMI until the operator performs a fresh field calibration via the `CALIBRATE` action.

`replace_membrane()` additionally resets `power_on_time`, triggering a warm-up cycle (new PTFE membrane must polarise). `replace_reagent()` sets `skip_warmup=True` in the record since the DPD optics are already at operating temperature.

## Testing

Each sensor includes a validation function:

```python
from sensors import validate_pH_sensor

validate_pH_sensor()  # Should print "✓ pH sensor validation passed"
```

Run all tests via individual sensor files or the package.

## Design Principles

- Strict type hints and runtime validation
- Bounded memory usage (circular buffers)
- Explicit error handling
- Physical bounds enforcement

## References

- Mettler Toledo pH Measurement Guide
- Hach DR900 Colorimeter Manual
- AWWA M12 "Instrumentation and Control"
- ISA RP60.6 "pH Sensors"

## License

MIT License

## Author

Guilherme F. G. Santos  
February 20, 2026

---