"""Operator, historian, and engineering-workstation semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .area_models import AREA_IDS
from .fields import ScalarField
from .supervisory import SupervisoryTwinRecord, build_reference_supervisory_records
from .supervisory import build_supervisory_records

AlarmState = str
OperatorActionKind = str
EngineeringEventKind = str
OperatingMode = str

VALID_ALARM_STATES = {
    "normal",
    "advisory_high",
    "advisory_low",
    "stale",
    "maintenance_context",
}
VALID_OPERATOR_ACTION_KINDS = {
    "setpoint_change",
    "mode_change",
    "alarm_acknowledgement",
}
VALID_ENGINEERING_EVENT_KINDS = {
    "configuration_review",
    "profile_validation",
    "metadata_export",
}
VALID_OPERATING_MODES = {"auto", "manual", "maintenance"}


@dataclass(frozen=True)
class HistorianTrendTag:
    tag_id: str
    area_id: str
    variable: str
    units: str
    source_record_id: str
    sampling_interval_seconds: float
    retention_class: str = "synthetic_short_window"
    evidence_status: str = "simulated_metadata"
    limitations: tuple[str, ...] = (
        "synthetic historian trend tag",
        "not operational historian evidence",
        "does not prove site coverage or historian integration",
    )

    def validate(self) -> None:
        if not self.tag_id:
            raise ValueError("tag_id is required")
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.tag_id}: unsupported area")
        if not self.variable:
            raise ValueError(f"{self.tag_id}: variable is required")
        if not self.units:
            raise ValueError(f"{self.tag_id}: units are required")
        if not self.source_record_id:
            raise ValueError(f"{self.tag_id}: source record is required")
        if self.sampling_interval_seconds <= 0.0:
            raise ValueError(f"{self.tag_id}: sampling interval must be positive")
        if not self.retention_class:
            raise ValueError(f"{self.tag_id}: retention class is required")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.tag_id}: evidence status must be simulated")
        if not self.limitations:
            raise ValueError(f"{self.tag_id}: limitations are required")


@dataclass(frozen=True)
class SupervisoryAlarmState:
    alarm_id: str
    area_id: str
    variable: str
    value: float
    state: AlarmState
    threshold_low: float | None
    threshold_high: float | None
    source_record_id: str
    priority: str = "review"
    evidence_status: str = "simulated_metadata"
    limitations: tuple[str, ...] = (
        "synthetic alarm state",
        "review prompt only",
        "not incident, safety, or operational alarm evidence",
    )

    def validate(self) -> None:
        if not self.alarm_id:
            raise ValueError("alarm_id is required")
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.alarm_id}: unsupported area")
        if not self.variable:
            raise ValueError(f"{self.alarm_id}: variable is required")
        if not math.isfinite(self.value):
            raise ValueError(f"{self.alarm_id}: value must be finite")
        if self.state not in VALID_ALARM_STATES:
            raise ValueError(f"{self.alarm_id}: unsupported alarm state")
        if self.threshold_low is not None and not math.isfinite(self.threshold_low):
            raise ValueError(f"{self.alarm_id}: low threshold must be finite")
        if self.threshold_high is not None and not math.isfinite(self.threshold_high):
            raise ValueError(f"{self.alarm_id}: high threshold must be finite")
        if not self.source_record_id:
            raise ValueError(f"{self.alarm_id}: source record is required")
        if self.priority not in {"info", "review"}:
            raise ValueError(f"{self.alarm_id}: unsupported priority")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.alarm_id}: evidence status must be simulated")
        if not self.limitations:
            raise ValueError(f"{self.alarm_id}: limitations are required")


@dataclass(frozen=True)
class MaintenanceWindow:
    window_id: str
    area_id: str
    start_ms: int
    end_ms: int
    purpose: str
    operating_mode: OperatingMode = "maintenance"
    evidence_status: str = "simulated_metadata"
    limitations: tuple[str, ...] = (
        "synthetic maintenance window",
        "not a real work order or maintenance authorization",
    )

    def validate(self) -> None:
        if not self.window_id:
            raise ValueError("window_id is required")
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.window_id}: unsupported area")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError(f"{self.window_id}: invalid time window")
        if not self.purpose:
            raise ValueError(f"{self.window_id}: purpose is required")
        if self.operating_mode not in VALID_OPERATING_MODES:
            raise ValueError(f"{self.window_id}: unsupported operating mode")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.window_id}: evidence status must be simulated")
        if not self.limitations:
            raise ValueError(f"{self.window_id}: limitations are required")


@dataclass(frozen=True)
class OperatorActionRecord:
    action_id: str
    area_id: str
    action_kind: OperatorActionKind
    variable: str
    previous_value: float
    requested_value: float
    units: str
    operating_mode: OperatingMode
    maintenance_window_id: str = ""
    evidence_status: str = "simulated_metadata"
    limitations: tuple[str, ...] = (
        "synthetic operator action",
        "not a real operator action or authorization record",
    )

    def validate(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.action_id}: unsupported area")
        if self.action_kind not in VALID_OPERATOR_ACTION_KINDS:
            raise ValueError(f"{self.action_id}: unsupported action kind")
        if not self.variable:
            raise ValueError(f"{self.action_id}: variable is required")
        if not math.isfinite(self.previous_value) or not math.isfinite(
            self.requested_value
        ):
            raise ValueError(f"{self.action_id}: values must be finite")
        if not self.units:
            raise ValueError(f"{self.action_id}: units are required")
        if self.operating_mode not in VALID_OPERATING_MODES:
            raise ValueError(f"{self.action_id}: unsupported operating mode")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.action_id}: evidence status must be simulated")
        if not self.limitations:
            raise ValueError(f"{self.action_id}: limitations are required")


@dataclass(frozen=True)
class EngineeringWorkstationEvent:
    event_id: str
    area_id: str
    event_kind: EngineeringEventKind
    target: str
    operating_mode: OperatingMode
    source_profile_id: str
    evidence_status: str = "simulated_metadata"
    limitations: tuple[str, ...] = (
        "synthetic engineering-workstation event",
        "not a real configuration change or vendor engineering tool event",
    )

    def validate(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.event_id}: unsupported area")
        if self.event_kind not in VALID_ENGINEERING_EVENT_KINDS:
            raise ValueError(f"{self.event_id}: unsupported event kind")
        if not self.target:
            raise ValueError(f"{self.event_id}: target is required")
        if self.operating_mode not in VALID_OPERATING_MODES:
            raise ValueError(f"{self.event_id}: unsupported operating mode")
        if not self.source_profile_id:
            raise ValueError(f"{self.event_id}: source profile is required")
        if self.evidence_status != "simulated_metadata":
            raise ValueError(f"{self.event_id}: evidence status must be simulated")
        if not self.limitations:
            raise ValueError(f"{self.event_id}: limitations are required")


@dataclass(frozen=True)
class OperatorHistorianSemanticBundle:
    area_id: str
    trend_tags: tuple[HistorianTrendTag, ...]
    alarm_states: tuple[SupervisoryAlarmState, ...]
    maintenance_windows: tuple[MaintenanceWindow, ...]
    operator_actions: tuple[OperatorActionRecord, ...]
    engineering_events: tuple[EngineeringWorkstationEvent, ...]
    evidence_status: str = "simulated_metadata"

    def validate(self) -> None:
        if self.area_id not in AREA_IDS:
            raise ValueError("unsupported area")
        if self.evidence_status != "simulated_metadata":
            raise ValueError("bundle evidence status must be simulated")
        for collection in (
            self.trend_tags,
            self.alarm_states,
            self.maintenance_windows,
            self.operator_actions,
            self.engineering_events,
        ):
            if not collection:
                raise ValueError("semantic bundle collections must not be empty")
            for item in collection:
                item.validate()


def build_historian_trend_tags(
    records: tuple[SupervisoryTwinRecord, ...],
    *,
    sampling_interval_seconds: float = 5.0,
) -> tuple[HistorianTrendTag, ...]:
    if sampling_interval_seconds <= 0.0:
        raise ValueError("sampling interval must be positive")
    tags: list[HistorianTrendTag] = []
    for record in records:
        record.validate()
        if record.role != "historian":
            continue
        tag = HistorianTrendTag(
            tag_id=f"TAG-{record.area_id.upper()}-{record.variable.upper()}",
            area_id=record.area_id,
            variable=record.variable,
            units=record.units,
            source_record_id=record.record_id,
            sampling_interval_seconds=sampling_interval_seconds,
        )
        tag.validate()
        tags.append(tag)
    return tuple(tags)


def derive_alarm_state(
    record: SupervisoryTwinRecord,
    *,
    threshold_low: float | None = None,
    threshold_high: float | None = None,
) -> SupervisoryAlarmState:
    record.validate()
    state = "normal"
    if threshold_low is not None and record.value < threshold_low:
        state = "advisory_low"
    if threshold_high is not None and record.value > threshold_high:
        state = "advisory_high"
    alarm = SupervisoryAlarmState(
        alarm_id=f"ALM-{record.area_id.upper()}-{record.variable.upper()}",
        area_id=record.area_id,
        variable=record.variable,
        value=record.value,
        state=state,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        source_record_id=record.record_id,
        priority="review" if state != "normal" else "info",
    )
    alarm.validate()
    return alarm


def build_reference_operator_historian_semantics(
    area_id: str,
) -> OperatorHistorianSemanticBundle:
    from .area_models import reference_area_model

    model = reference_area_model(area_id)
    scalar = ScalarField.uniform(
        model.mesh,
        name=model.scalar_names[0],
        units="process_units",
        value=0.0,
    )
    records = build_supervisory_records(model, scalar=scalar) + (
        build_reference_supervisory_records(area_id)
    )
    trend_tags = build_historian_trend_tags(records)
    first_record = records[0]
    alarm_states = (
        derive_alarm_state(first_record, threshold_low=0.0, threshold_high=1000.0),
    )
    maintenance = MaintenanceWindow(
        window_id=f"MW-{area_id.upper()}-001",
        area_id=area_id,
        start_ms=0,
        end_ms=900000,
        purpose="synthetic operator/historian semantics review window",
    )
    operator_action = OperatorActionRecord(
        action_id=f"OP-{area_id.upper()}-SETPOINT-001",
        area_id=area_id,
        action_kind="setpoint_change",
        variable=first_record.variable,
        previous_value=first_record.value,
        requested_value=first_record.value,
        units=first_record.units,
        operating_mode="maintenance",
        maintenance_window_id=maintenance.window_id,
    )
    engineering_event = EngineeringWorkstationEvent(
        event_id=f"ENG-{area_id.upper()}-PROFILE-001",
        area_id=area_id,
        event_kind="profile_validation",
        target="synthetic supervisory profile metadata",
        operating_mode="maintenance",
        source_profile_id="engineering-model-context",
    )
    bundle = OperatorHistorianSemanticBundle(
        area_id=area_id,
        trend_tags=trend_tags,
        alarm_states=alarm_states,
        maintenance_windows=(maintenance,),
        operator_actions=(operator_action,),
        engineering_events=(engineering_event,),
    )
    bundle.validate()
    return bundle
