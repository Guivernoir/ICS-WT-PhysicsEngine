"""Digital-twin metadata, calibration, and uncertainty records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TwinStatus(StrEnum):
    SYNTHETIC_UNVALIDATED = "synthetic_unvalidated"
    SYNTHETIC_VERIFIED = "synthetic_verified"
    CALIBRATION_READY = "calibration_ready"
    EVIDENCE_CALIBRATED = "evidence_calibrated"


class CalibrationEvidenceClass(StrEnum):
    SYNTHETIC_REFERENCE = "synthetic_reference"
    MANUFACTURED_SOLUTION = "manufactured_solution"
    LAB_MEASUREMENT = "lab_measurement"
    FIELD_TELEMETRY = "field_telemetry"
    EXTERNAL_CFD_COMPARISON = "external_cfd_comparison"


class CalibrationConfidence(StrEnum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CalibrationStatus(StrEnum):
    UNCALIBRATED = "uncalibrated"
    CALIBRATION_READY = "calibration_ready"
    SYNTHETIC_CALIBRATED = "synthetic_calibrated"
    EVIDENCE_CALIBRATED = "evidence_calibrated"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CalibrationRecord:
    parameter: str
    source: str
    value: float
    units: str
    status: TwinStatus

    def validate(self) -> None:
        if not self.parameter:
            raise ValueError("calibration parameter is required")
        if self.status == TwinStatus.EVIDENCE_CALIBRATED and not self.source:
            raise ValueError("evidence-calibrated parameters require a source")


@dataclass(frozen=True)
class CalibrationEvidenceSource:
    source_id: str
    evidence_class: CalibrationEvidenceClass
    description: str
    source_reference: str = ""
    limitations: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.source_id:
            raise ValueError("calibration source_id is required")
        if not self.description:
            raise ValueError(f"{self.source_id}: description is required")
        if not self.limitations:
            raise ValueError(f"{self.source_id}: limitations are required")
        non_synthetic = {
            CalibrationEvidenceClass.LAB_MEASUREMENT,
            CalibrationEvidenceClass.FIELD_TELEMETRY,
            CalibrationEvidenceClass.EXTERNAL_CFD_COMPARISON,
        }
        if self.evidence_class in non_synthetic and not self.source_reference:
            raise ValueError(
                f"{self.source_id}: non-synthetic evidence requires a reference"
            )


@dataclass(frozen=True)
class CalibrationParameter:
    parameter: str
    value: float
    units: str
    source_id: str
    accepted_min: float | None = None
    accepted_max: float | None = None

    def validate(self) -> None:
        if not self.parameter:
            raise ValueError("calibration parameter name is required")
        if not self.units:
            raise ValueError(f"{self.parameter}: units are required")
        if not self.source_id:
            raise ValueError(f"{self.parameter}: source_id is required")
        if (
            self.accepted_min is not None
            and self.accepted_max is not None
            and self.accepted_min > self.accepted_max
        ):
            raise ValueError(f"{self.parameter}: invalid accepted range")
        if self.accepted_min is not None and self.value < self.accepted_min:
            raise ValueError(f"{self.parameter}: value below accepted minimum")
        if self.accepted_max is not None and self.value > self.accepted_max:
            raise ValueError(f"{self.parameter}: value above accepted maximum")


@dataclass(frozen=True)
class CalibrationResidual:
    parameter: str
    observed: float
    predicted: float
    tolerance: float
    absolute_residual: float
    accepted: bool

    @classmethod
    def from_values(
        cls, *, parameter: str, observed: float, predicted: float, tolerance: float
    ) -> "CalibrationResidual":
        residual, accepted = evaluate_calibration_residual(
            observed=observed,
            predicted=predicted,
            tolerance=tolerance,
        )
        return cls(
            parameter=parameter,
            observed=observed,
            predicted=predicted,
            tolerance=tolerance,
            absolute_residual=residual,
            accepted=accepted,
        )

    def validate(self) -> None:
        if not self.parameter:
            raise ValueError("residual parameter is required")
        if self.tolerance < 0.0:
            raise ValueError(f"{self.parameter}: tolerance cannot be negative")
        if self.absolute_residual < 0.0:
            raise ValueError(f"{self.parameter}: residual cannot be negative")
        expected = abs(self.observed - self.predicted)
        if abs(expected - self.absolute_residual) > 1.0e-12:
            raise ValueError(f"{self.parameter}: residual does not match values")
        if self.accepted != (self.absolute_residual <= self.tolerance):
            raise ValueError(f"{self.parameter}: accepted flag is inconsistent")


@dataclass(frozen=True)
class UncertaintyRecord:
    quantity: str
    absolute_bound: float
    basis: str

    def validate(self) -> None:
        if not self.quantity:
            raise ValueError("uncertainty quantity is required")
        if self.absolute_bound < 0.0:
            raise ValueError("uncertainty bound cannot be negative")
        if not self.basis:
            raise ValueError("uncertainty basis is required")


@dataclass(frozen=True)
class DigitalTwinMetadata:
    name: str
    geometry_reference: str
    equipment_reference: str
    status: TwinStatus
    calibration: tuple[CalibrationRecord, ...] = ()
    uncertainty: tuple[UncertaintyRecord, ...] = ()

    def validate(self) -> None:
        if not self.name:
            raise ValueError("digital twin name is required")
        if not self.geometry_reference:
            raise ValueError("geometry reference is required")
        if not self.equipment_reference:
            raise ValueError("equipment reference is required")
        for calibration_record in self.calibration:
            calibration_record.validate()
        for uncertainty_record in self.uncertainty:
            uncertainty_record.validate()


@dataclass(frozen=True)
class CalibrationAssessment:
    area_id: str
    calibration_status: CalibrationStatus
    twin_status: TwinStatus
    confidence: CalibrationConfidence
    sources: tuple[CalibrationEvidenceSource, ...]
    parameters: tuple[CalibrationParameter, ...]
    residuals: tuple[CalibrationResidual, ...]
    uncertainty: tuple[UncertaintyRecord, ...]
    caveats: tuple[str, ...]
    calibration_records: tuple[CalibrationRecord, ...] = ()

    def validate(self) -> None:
        if not self.area_id:
            raise ValueError("calibration assessment area_id is required")
        if not self.caveats:
            raise ValueError(f"{self.area_id}: calibration caveats are required")
        source_ids = set()
        for source in self.sources:
            source.validate()
            source_ids.add(source.source_id)
        for parameter in self.parameters:
            parameter.validate()
            if parameter.source_id not in source_ids:
                raise ValueError(
                    f"{parameter.parameter}: source_id is not present in sources"
                )
        for residual in self.residuals:
            residual.validate()
        for uncertainty_record in self.uncertainty:
            uncertainty_record.validate()
        for calibration_record in self.calibration_records:
            calibration_record.validate()
        if self.calibration_status == CalibrationStatus.EVIDENCE_CALIBRATED:
            if self.twin_status != TwinStatus.EVIDENCE_CALIBRATED:
                raise ValueError("evidence-calibrated assessment has wrong twin status")
            if not any(
                record.status == TwinStatus.EVIDENCE_CALIBRATED
                for record in self.calibration_records
            ):
                raise ValueError(
                    "evidence-calibrated assessment requires a calibration record"
                )
            if "not real-plant validation" not in " ".join(self.caveats):
                raise ValueError(
                    "evidence-calibrated assessment must preserve validation caveat"
                )
        if self.calibration_status == CalibrationStatus.SYNTHETIC_CALIBRATED:
            if self.twin_status != TwinStatus.SYNTHETIC_VERIFIED:
                raise ValueError(
                    "synthetic-calibrated assessment has wrong twin status"
                )


def _all_sources_are_synthetic(
    sources: tuple[CalibrationEvidenceSource, ...],
) -> bool:
    synthetic_classes = {
        CalibrationEvidenceClass.SYNTHETIC_REFERENCE,
        CalibrationEvidenceClass.MANUFACTURED_SOLUTION,
    }
    return all(source.evidence_class in synthetic_classes for source in sources)


def assess_calibration_evidence(
    *,
    area_id: str,
    sources: tuple[CalibrationEvidenceSource, ...] = (),
    parameters: tuple[CalibrationParameter, ...] = (),
    residuals: tuple[CalibrationResidual, ...] = (),
    uncertainty: tuple[UncertaintyRecord, ...] = (),
    calibration_records: tuple[CalibrationRecord, ...] = (),
) -> CalibrationAssessment:
    """Classify calibration evidence without upgrading validation status."""

    if not sources or not parameters:
        assessment = CalibrationAssessment(
            area_id=area_id,
            calibration_status=CalibrationStatus.UNCALIBRATED,
            twin_status=TwinStatus.SYNTHETIC_UNVALIDATED,
            confidence=CalibrationConfidence.UNKNOWN,
            sources=sources,
            parameters=parameters,
            residuals=residuals,
            uncertainty=uncertainty,
            calibration_records=calibration_records,
            caveats=(
                "uncalibrated synthetic model",
                "not certification, commissioning, safety, or design authority",
            ),
        )
        assessment.validate()
        return assessment

    if not residuals:
        assessment = CalibrationAssessment(
            area_id=area_id,
            calibration_status=CalibrationStatus.CALIBRATION_READY,
            twin_status=TwinStatus.CALIBRATION_READY,
            confidence=CalibrationConfidence.LOW,
            sources=sources,
            parameters=parameters,
            residuals=residuals,
            uncertainty=uncertainty,
            calibration_records=calibration_records,
            caveats=(
                "parameters are present but residual evidence is missing",
                "not real-plant validation or commissioning evidence",
            ),
        )
        assessment.validate()
        return assessment

    if any(not residual.accepted for residual in residuals):
        assessment = CalibrationAssessment(
            area_id=area_id,
            calibration_status=CalibrationStatus.REJECTED,
            twin_status=TwinStatus.SYNTHETIC_UNVALIDATED,
            confidence=CalibrationConfidence.LOW,
            sources=sources,
            parameters=parameters,
            residuals=residuals,
            uncertainty=uncertainty,
            calibration_records=calibration_records,
            caveats=(
                "one or more residuals exceed tolerance",
                "rejected calibration cannot support stronger model claims",
            ),
        )
        assessment.validate()
        return assessment

    if _all_sources_are_synthetic(sources):
        assessment = CalibrationAssessment(
            area_id=area_id,
            calibration_status=CalibrationStatus.SYNTHETIC_CALIBRATED,
            twin_status=TwinStatus.SYNTHETIC_VERIFIED,
            confidence=CalibrationConfidence.MEDIUM,
            sources=sources,
            parameters=parameters,
            residuals=residuals,
            uncertainty=uncertainty,
            calibration_records=calibration_records,
            caveats=(
                "synthetic calibration evidence only",
                "not real-plant validation or commissioning evidence",
            ),
        )
        assessment.validate()
        return assessment

    has_evidence_record = any(
        record.status == TwinStatus.EVIDENCE_CALIBRATED
        for record in calibration_records
    )
    if not has_evidence_record:
        assessment = CalibrationAssessment(
            area_id=area_id,
            calibration_status=CalibrationStatus.CALIBRATION_READY,
            twin_status=TwinStatus.CALIBRATION_READY,
            confidence=CalibrationConfidence.MEDIUM,
            sources=sources,
            parameters=parameters,
            residuals=residuals,
            uncertainty=uncertainty,
            calibration_records=calibration_records,
            caveats=(
                "accepted residuals require explicit calibration record",
                "not real-plant validation or commissioning evidence",
            ),
        )
        assessment.validate()
        return assessment

    assessment = CalibrationAssessment(
        area_id=area_id,
        calibration_status=CalibrationStatus.EVIDENCE_CALIBRATED,
        twin_status=TwinStatus.EVIDENCE_CALIBRATED,
        confidence=CalibrationConfidence.HIGH,
        sources=sources,
        parameters=parameters,
        residuals=residuals,
        uncertainty=uncertainty,
        calibration_records=calibration_records,
        caveats=(
            "evidence-calibrated parameter set",
            "not real-plant validation or safety-system proof",
        ),
    )
    assessment.validate()
    return assessment


def build_reference_calibration_assessment(
    area_id: str,
    uncertainty: tuple[UncertaintyRecord, ...] = (),
) -> CalibrationAssessment:
    """Return the default uncalibrated assessment for a reference area."""

    return assess_calibration_evidence(area_id=area_id, uncertainty=uncertainty)


def evaluate_calibration_residual(
    *, observed: float, predicted: float, tolerance: float
) -> tuple[float, bool]:
    """Return absolute residual and whether it fits the supplied tolerance."""

    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    residual = abs(observed - predicted)
    return residual, residual <= tolerance
