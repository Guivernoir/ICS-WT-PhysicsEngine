"""Calibration evidence fitting separated from validation status."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .digital_twin import (
    CalibrationAssessment,
    CalibrationEvidenceClass,
    CalibrationEvidenceSource,
    CalibrationParameter,
    CalibrationRecord,
    CalibrationResidual,
    TwinStatus,
    UncertaintyRecord,
    assess_calibration_evidence,
)


@dataclass(frozen=True)
class CalibrationDataPoint:
    sample_id: str
    source_id: str
    parameter: str
    observed: float
    simulated: float
    sensitivity: float = 1.0
    weight: float = 1.0

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("calibration sample_id is required")
        if not self.source_id:
            raise ValueError(f"{self.sample_id}: source_id is required")
        if not self.parameter:
            raise ValueError(f"{self.sample_id}: parameter is required")
        values = (self.observed, self.simulated, self.sensitivity, self.weight)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{self.sample_id}: values must be finite")
        if self.sensitivity == 0.0:
            raise ValueError(f"{self.sample_id}: sensitivity cannot be zero")
        if self.weight <= 0.0:
            raise ValueError(f"{self.sample_id}: weight must be positive")


@dataclass(frozen=True)
class RejectedCalibrationData:
    sample_id: str
    source_id: str
    reason: str

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("rejected sample_id is required")
        if not self.source_id:
            raise ValueError(f"{self.sample_id}: source_id is required")
        if not self.reason:
            raise ValueError(f"{self.sample_id}: rejection reason is required")


@dataclass(frozen=True)
class CalibrationEvidenceFit:
    area_id: str
    source: CalibrationEvidenceSource
    fitted_parameter: CalibrationParameter
    residuals: tuple[CalibrationResidual, ...]
    rejected_data: tuple[RejectedCalibrationData, ...]
    uncertainty_update: UncertaintyRecord
    calibration_record: CalibrationRecord
    validation_status: str
    caveats: tuple[str, ...]

    def validate(self) -> None:
        if not self.area_id:
            raise ValueError("calibration fit area_id is required")
        self.source.validate()
        self.fitted_parameter.validate()
        if self.fitted_parameter.source_id != self.source.source_id:
            raise ValueError("fitted parameter source does not match evidence source")
        if not self.residuals:
            raise ValueError("calibration fit residuals are required")
        for residual in self.residuals:
            residual.validate()
        for rejected in self.rejected_data:
            rejected.validate()
        self.uncertainty_update.validate()
        self.calibration_record.validate()
        if self.calibration_record.parameter != self.fitted_parameter.parameter:
            raise ValueError("calibration record parameter does not match fit")
        if self.calibration_record.source != self.source.source_id:
            raise ValueError("calibration record source does not match fit")
        if self.validation_status != "fitted_not_validated":
            raise ValueError("calibration fit must not claim validation")
        if "not validation" not in " ".join(self.caveats):
            raise ValueError("calibration fit must preserve validation caveat")


def _is_synthetic_source(source: CalibrationEvidenceSource) -> bool:
    return source.evidence_class in {
        CalibrationEvidenceClass.SYNTHETIC_REFERENCE,
        CalibrationEvidenceClass.MANUFACTURED_SOLUTION,
    }


def _record_status_for_source(source: CalibrationEvidenceSource) -> TwinStatus:
    if _is_synthetic_source(source):
        return TwinStatus.SYNTHETIC_VERIFIED
    return TwinStatus.EVIDENCE_CALIBRATED


def fit_calibration_parameter(
    *,
    area_id: str,
    source: CalibrationEvidenceSource,
    parameter: str,
    units: str,
    samples: tuple[CalibrationDataPoint, ...],
    residual_tolerance: float,
    accepted_min: float | None = None,
    accepted_max: float | None = None,
) -> CalibrationEvidenceFit:
    """Fit a one-parameter correction while preserving validation separation."""

    if residual_tolerance < 0.0:
        raise ValueError("residual_tolerance cannot be negative")
    source.validate()
    accepted_samples: list[CalibrationDataPoint] = []
    rejected: list[RejectedCalibrationData] = []
    for sample in samples:
        try:
            sample.validate()
        except ValueError as exc:
            rejected.append(
                RejectedCalibrationData(
                    sample_id=sample.sample_id or "unknown",
                    source_id=sample.source_id or "unknown",
                    reason=str(exc),
                )
            )
            continue
        if sample.source_id != source.source_id:
            rejected.append(
                RejectedCalibrationData(
                    sample_id=sample.sample_id,
                    source_id=sample.source_id,
                    reason="sample source does not match calibration source",
                )
            )
            continue
        if sample.parameter != parameter:
            rejected.append(
                RejectedCalibrationData(
                    sample_id=sample.sample_id,
                    source_id=sample.source_id,
                    reason="sample parameter does not match fit parameter",
                )
            )
            continue
        accepted_samples.append(sample)
    if not accepted_samples:
        raise ValueError("at least one accepted calibration sample is required")

    weighted_correction = 0.0
    total_weight = 0.0
    for sample in accepted_samples:
        weighted_correction += (
            (sample.observed - sample.simulated) / sample.sensitivity
        ) * sample.weight
        total_weight += sample.weight
    fitted_value = weighted_correction / total_weight
    fitted_parameter = CalibrationParameter(
        parameter=parameter,
        value=fitted_value,
        units=units,
        source_id=source.source_id,
        accepted_min=accepted_min,
        accepted_max=accepted_max,
    )
    fitted_parameter.validate()

    residuals = tuple(
        CalibrationResidual.from_values(
            parameter=parameter,
            observed=sample.observed,
            predicted=sample.simulated + (sample.sensitivity * fitted_value),
            tolerance=residual_tolerance,
        )
        for sample in accepted_samples
    )
    uncertainty_bound = max(residual.absolute_residual for residual in residuals)
    uncertainty_update = UncertaintyRecord(
        quantity=f"{parameter}_fit_residual_bound",
        absolute_bound=uncertainty_bound,
        basis="HS-30B accepted calibration residual envelope",
    )
    calibration_record = CalibrationRecord(
        parameter=parameter,
        source=source.source_id,
        value=fitted_value,
        units=units,
        status=_record_status_for_source(source),
    )
    fit = CalibrationEvidenceFit(
        area_id=area_id,
        source=source,
        fitted_parameter=fitted_parameter,
        residuals=residuals,
        rejected_data=tuple(rejected),
        uncertainty_update=uncertainty_update,
        calibration_record=calibration_record,
        validation_status="fitted_not_validated",
        caveats=(
            "calibration parameter fitting is not validation",
            "not real-plant validation, certification, or commissioning evidence",
        ),
    )
    fit.validate()
    return fit


def assess_calibration_fit(fit: CalibrationEvidenceFit) -> CalibrationAssessment:
    """Convert a fitted parameter into the caveated assessment model."""

    fit.validate()
    return assess_calibration_evidence(
        area_id=fit.area_id,
        sources=(fit.source,),
        parameters=(fit.fitted_parameter,),
        residuals=fit.residuals,
        uncertainty=(fit.uncertainty_update,),
        calibration_records=(fit.calibration_record,),
    )
