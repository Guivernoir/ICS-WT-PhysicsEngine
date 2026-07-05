"""Digital-twin validation gate records with no-overclaim boundaries."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from ..area_models import AREA_IDS, reference_area_model
from ..digital_twin import (
    CalibrationConfidence,
    CalibrationStatus,
    TwinStatus,
    build_reference_calibration_assessment,
)
from ..verification import (
    NumericalVerificationSuite,
    build_reference_numerical_verification_suite,
)

REAL_PLANT_VALIDATION_BLOCKED = (
    "blocked_missing_real_calibration_and_external_validation"
)
EXTERNAL_COMPARISON_NOT_VALIDATION = (
    "blocked_external_comparison_not_real_plant_validation"
)


@dataclass(frozen=True)
class ExternalCfdComparisonRecord:
    comparison_id: str
    area_id: str
    source_reference: str
    metric: str
    hydrasim_value: float
    external_value: float
    tolerance: float
    accepted: bool
    evidence_status: str = "external_cfd_comparison_record"
    limitations: tuple[str, ...] = (
        "external CFD comparison is not real-plant validation by itself",
    )

    def validate(self) -> None:
        if not self.comparison_id:
            raise ValueError("comparison_id is required")
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.comparison_id}: unsupported area_id")
        if not self.source_reference:
            raise ValueError(f"{self.comparison_id}: source_reference is required")
        if not self.metric:
            raise ValueError(f"{self.comparison_id}: metric is required")
        values = (self.hydrasim_value, self.external_value, self.tolerance)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{self.comparison_id}: values must be finite")
        if self.tolerance < 0.0:
            raise ValueError(f"{self.comparison_id}: tolerance cannot be negative")
        expected = abs(self.hydrasim_value - self.external_value) <= self.tolerance
        if self.accepted != expected:
            raise ValueError(f"{self.comparison_id}: accepted flag is inconsistent")
        if self.evidence_status != "external_cfd_comparison_record":
            raise ValueError(f"{self.comparison_id}: unsupported evidence status")
        if "not real-plant validation" not in " ".join(self.limitations):
            raise ValueError(f"{self.comparison_id}: missing validation caveat")


@dataclass(frozen=True)
class DigitalTwinValidationAreaRecord:
    area_id: str
    model_name: str
    twin_status: TwinStatus
    calibration_status: CalibrationStatus
    calibration_confidence: CalibrationConfidence
    reference_model_checked: bool
    evidence_status: str
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if self.area_id not in AREA_IDS:
            raise ValueError(f"{self.area_id}: unsupported validation area")
        if not self.model_name:
            raise ValueError(f"{self.area_id}: model_name is required")
        if not self.reference_model_checked:
            raise ValueError(f"{self.area_id}: reference model check is required")
        if self.evidence_status != "synthetic_digital_twin_validation_area":
            raise ValueError(f"{self.area_id}: unsupported evidence status")
        if "not real-plant validation" not in " ".join(self.limitations):
            raise ValueError(f"{self.area_id}: missing validation caveat")


@dataclass(frozen=True)
class DigitalTwinValidationGate:
    gate_id: str
    numerical_suite: NumericalVerificationSuite
    area_records: tuple[DigitalTwinValidationAreaRecord, ...]
    external_comparisons: tuple[ExternalCfdComparisonRecord, ...]
    implementation_verified: bool
    real_plant_validation_status: str
    evidence_status: str
    limitations: tuple[str, ...]

    def validate(self) -> None:
        if not self.gate_id:
            raise ValueError("validation gate_id is required")
        self.numerical_suite.validate()
        if not self.area_records:
            raise ValueError(f"{self.gate_id}: area records are required")
        for record in self.area_records:
            record.validate()
        for comparison in self.external_comparisons:
            comparison.validate()
        expected_implementation = self.numerical_suite.passed and all(
            record.reference_model_checked for record in self.area_records
        )
        if self.implementation_verified != expected_implementation:
            raise ValueError(f"{self.gate_id}: implementation flag is inconsistent")
        accepted_external = any(
            comparison.accepted for comparison in self.external_comparisons
        )
        expected_status = (
            EXTERNAL_COMPARISON_NOT_VALIDATION
            if accepted_external
            else REAL_PLANT_VALIDATION_BLOCKED
        )
        if self.real_plant_validation_status != expected_status:
            raise ValueError(f"{self.gate_id}: validation status is inconsistent")
        if self.evidence_status != "synthetic_digital_twin_validation_gate":
            raise ValueError(f"{self.gate_id}: unsupported evidence status")
        if "not real-plant validation" not in " ".join(self.limitations):
            raise ValueError(f"{self.gate_id}: missing validation caveat")


def build_reference_validation_area_records() -> (
    tuple[DigitalTwinValidationAreaRecord, ...]
):
    records: list[DigitalTwinValidationAreaRecord] = []
    for area_id in AREA_IDS:
        model = reference_area_model(area_id)
        model.twin_metadata.validate()
        assessment = build_reference_calibration_assessment(
            area_id,
            uncertainty=model.twin_metadata.uncertainty,
        )
        assessment.validate()
        record = DigitalTwinValidationAreaRecord(
            area_id=area_id,
            model_name=model.twin_metadata.name,
            twin_status=assessment.twin_status,
            calibration_status=assessment.calibration_status,
            calibration_confidence=assessment.confidence,
            reference_model_checked=True,
            evidence_status="synthetic_digital_twin_validation_area",
            limitations=(
                "reference area model is synthetically checked",
                "not real-plant validation, certification, or commissioning evidence",
            ),
        )
        record.validate()
        records.append(record)
    return tuple(records)


def build_digital_twin_validation_gate(
    external_comparisons: tuple[ExternalCfdComparisonRecord, ...] = (),
) -> DigitalTwinValidationGate:
    suite = build_reference_numerical_verification_suite()
    area_records = build_reference_validation_area_records()
    for comparison in external_comparisons:
        comparison.validate()
    accepted_external = any(comparison.accepted for comparison in external_comparisons)
    gate = DigitalTwinValidationGate(
        gate_id="hs-34-digital-twin-validation-gate",
        numerical_suite=suite,
        area_records=area_records,
        external_comparisons=external_comparisons,
        implementation_verified=suite.passed
        and all(record.reference_model_checked for record in area_records),
        real_plant_validation_status=(
            EXTERNAL_COMPARISON_NOT_VALIDATION
            if accepted_external
            else REAL_PLANT_VALIDATION_BLOCKED
        ),
        evidence_status="synthetic_digital_twin_validation_gate",
        limitations=(
            "implementation verification combines numerical checks and model records",
            "not real-plant validation, certification, commissioning, or safety evidence",
            "real-plant validation remains blocked without calibration and validation data",
        ),
    )
    gate.validate()
    return gate


def _comparison_payload(comparison: ExternalCfdComparisonRecord) -> dict[str, object]:
    payload = asdict(comparison)
    payload["limitations"] = list(comparison.limitations)
    return payload


def _area_payload(record: DigitalTwinValidationAreaRecord) -> dict[str, object]:
    return {
        "area_id": record.area_id,
        "model_name": record.model_name,
        "twin_status": record.twin_status.value,
        "calibration_status": record.calibration_status.value,
        "calibration_confidence": record.calibration_confidence.value,
        "reference_model_checked": record.reference_model_checked,
        "evidence_status": record.evidence_status,
        "limitations": list(record.limitations),
    }


def render_digital_twin_validation_gate_json(
    gate: DigitalTwinValidationGate,
) -> str:
    gate.validate()
    payload = {
        "gate_id": gate.gate_id,
        "evidence_status": gate.evidence_status,
        "implementation_verified": gate.implementation_verified,
        "real_plant_validation_status": gate.real_plant_validation_status,
        "numerical_suite": {
            "suite_id": gate.numerical_suite.suite_id,
            "passed": gate.numerical_suite.passed,
            "case_count": len(gate.numerical_suite.results),
            "limitations": list(gate.numerical_suite.limitations),
        },
        "area_records": [_area_payload(record) for record in gate.area_records],
        "external_comparisons": [
            _comparison_payload(comparison) for comparison in gate.external_comparisons
        ],
        "limitations": list(gate.limitations),
    }
    return json.dumps(payload, indent=2, sort_keys=True)
