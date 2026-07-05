"""HydraSim project quality checks."""

from __future__ import annotations

import os
import re
import sys
import importlib.util
from pathlib import Path

from quality_hs import check_hs_docs

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MAX_FILES_PER_FOLDER = 20
MAX_LINES = 500
CODE_FILE_SUFFIXES = {".py", ".pyi"}
SKIP_DIRS = {
    ".git",
    ".private",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}
LOCAL_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _project_files() -> list[Path]:
    files: list[Path] = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        for name in names:
            path = Path(base) / name
            files.append(path)
    return files


def _check_folder_density(errors: list[str]) -> None:
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        visible = [name for name in names if not name.startswith(".")]
        if len(visible) > MAX_FILES_PER_FOLDER:
            errors.append(
                f"{_rel(Path(base))}: {len(visible)} files exceeds "
                f"{MAX_FILES_PER_FOLDER}; add subfolders"
            )


def _check_code_line_counts(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix not in CODE_FILE_SUFFIXES:
            continue
        rel = _rel(path)
        lines = path.read_text("utf-8").splitlines()
        if len(lines) > MAX_LINES:
            errors.append(
                f"{rel}: {len(lines)} lines exceeds hard {MAX_LINES}-line "
                "code-file limit; split the module"
            )


def _target_exists(source: Path, target: str) -> bool:
    clean = target.split("#", 1)[0].strip()
    if not clean or clean.startswith(("http://", "https://", "mailto:")):
        return True
    if clean.startswith("<") and clean.endswith(">"):
        clean = clean[1:-1]
    candidate = (source.parent / clean).resolve()
    return candidate.exists()


def _check_markdown_links(files: list[Path], errors: list[str]) -> None:
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text("utf-8")
        for match in LOCAL_LINK.finditer(text):
            target = match.group(1)
            if not _target_exists(path, target):
                errors.append(f"{_rel(path)}: broken local link {target!r}")


def _check_scenario_docs(errors: list[str]) -> None:
    from hydrasim.scenarios import get_scenario, scenario_ids, validate_scenario

    public_docs = (ROOT / "README.md").read_text("utf-8")
    for scenario_id in scenario_ids():
        scenario = get_scenario(scenario_id)
        problems = validate_scenario(scenario)
        if problems:
            errors.append(f"{scenario_id}: validation failed: {problems}")
        if scenario_id not in public_docs:
            errors.append(f"{scenario_id}: missing from public README")


def _check_cfd_docs(errors: list[str]) -> None:
    from hydrasim.hydraulics.cfd import (
        AREA_IDS,
        CalibrationStatus,
        TwinStatus,
        CalibrationDataPoint,
        CalibrationEvidenceClass,
        CalibrationEvidenceSource,
        CalibrationParameter,
        CalibrationResidual,
        assess_calibration_evidence,
        assess_calibration_fit,
        boundary_condition_catalog,
        build_digital_twin_validation_gate,
        build_external_review_calibration_gate,
        build_reference_calibration_assessment,
        build_reference_numerical_verification_suite,
        build_runtime_performance_gate,
        controller_coupling_catalog,
        device_coupling_catalog,
        export_cfd_summary,
        fit_calibration_parameter,
        performance_presets,
        render_digital_twin_validation_gate_json,
        render_external_review_calibration_gate_json,
        render_runtime_performance_gate_json,
        runtime_performance_budgets,
        build_reference_operator_historian_semantics,
        build_reference_supervisory_records,
        reference_area_model,
        supervisory_profile_catalog,
        unit_process_catalog,
    )

    public_docs = (ROOT / "README.md").read_text("utf-8")
    required_public_phrases = (
        "bounded CFD/digital-twin primitives",
        "Runtime Performance Gate",
        "Digital-Twin Validation Gate",
        "External Review And Calibration Evidence Gate",
        "CFD Lab Bundle v2",
        "Reference Water Plant CFD release-candidate",
        "synthetic",
        "certified design authority",
        "real-plant validation",
    )
    for phrase in required_public_phrases:
        if phrase not in public_docs:
            errors.append(f"public README: missing CFD phrase {phrase!r}")
    for area_id in AREA_IDS:
        if area_id not in public_docs:
            errors.append(f"{area_id}: missing from public README")
        model = reference_area_model(area_id)
        model.validate()
        assessment = build_reference_calibration_assessment(
            area_id,
            uncertainty=model.twin_metadata.uncertainty,
        )
        assessment.validate()
        if assessment.calibration_status != CalibrationStatus.UNCALIBRATED:
            errors.append(f"{area_id}: reference calibration is not uncalibrated")
        if assessment.twin_status != TwinStatus.SYNTHETIC_UNVALIDATED:
            errors.append(f"{area_id}: reference twin status is not unvalidated")
        first = export_cfd_summary(model)
        second = export_cfd_summary(model)
        if first != second:
            errors.append(f"{area_id}: CFD summary export is not deterministic")
    for unit in unit_process_catalog():
        unit.validate()
        limitation_text = " ".join(unit.limitations)
        if "not a certified design model" not in limitation_text:
            errors.append(f"{unit.unit_id}: missing no-overclaim limitation")
    for contract in boundary_condition_catalog():
        contract.validate()
        limitation_text = " ".join(contract.limitations)
        if (
            "not a commissioning or design-authority boundary model"
            not in limitation_text
        ):
            errors.append(f"{contract.condition_id}: missing no-overclaim limitation")
    for preset in performance_presets():
        preset.validate()
    for budget in runtime_performance_budgets():
        budget.validate()
    performance_records = build_runtime_performance_gate(iterations=1)
    for record in performance_records:
        record.validate()
        if not record.gate_passed:
            errors.append(f"{record.preset_id}: runtime performance gate failed")
    rendered_gate = render_runtime_performance_gate_json(performance_records)
    if rendered_gate != render_runtime_performance_gate_json(performance_records):
        errors.append("CFD runtime performance gate export is not deterministic")
    if "synthetic_runtime_performance_gate" not in rendered_gate:
        errors.append("CFD runtime performance gate: missing evidence status")
    validation_gate = build_digital_twin_validation_gate()
    validation_gate.validate()
    if not validation_gate.implementation_verified:
        errors.append("CFD validation gate: implementation is not verified")
    if (
        validation_gate.real_plant_validation_status
        != "blocked_missing_real_calibration_and_external_validation"
    ):
        errors.append("CFD validation gate: real-plant validation is not blocked")
    rendered_validation = render_digital_twin_validation_gate_json(validation_gate)
    if rendered_validation != render_digital_twin_validation_gate_json(validation_gate):
        errors.append("CFD validation gate export is not deterministic")
    if "synthetic_digital_twin_validation_gate" not in rendered_validation:
        errors.append("CFD validation gate: missing evidence status")
    review_gate = build_external_review_calibration_gate()
    review_gate.validate()
    if review_gate.evidence_disposition != "pending_external_review":
        errors.append("CFD review gate: default disposition is not pending")
    if review_gate.model_status_upgraded:
        errors.append("CFD review gate: model status upgraded automatically")
    rendered_review = render_external_review_calibration_gate_json(review_gate)
    if rendered_review != render_external_review_calibration_gate_json(review_gate):
        errors.append("CFD review gate export is not deterministic")
    if "synthetic_external_review_calibration_gate" not in rendered_review:
        errors.append("CFD review gate: missing evidence status")
    sensor_count = 0
    actuator_count = 0
    for contract in device_coupling_catalog():
        contract.validate()
        tag = getattr(contract, "sensor_tag", None) or getattr(
            contract, "actuator_tag", ""
        )
        if getattr(contract, "evidence_status", "") != "simulated_metadata":
            errors.append(f"{tag}: device coupling evidence status is not simulated")
        if hasattr(contract, "sensor_tag"):
            sensor_count += 1
        if hasattr(contract, "actuator_tag"):
            actuator_count += 1
    if sensor_count == 0:
        errors.append("CFD device coupling: missing sensor contracts")
    if actuator_count == 0:
        errors.append("CFD device coupling: missing actuator contracts")
    controller_count = 0
    for contract in controller_coupling_catalog():
        contract.validate()
        controller_count += 1
        if contract.evidence_status != "simulated_metadata":
            errors.append(
                f"{contract.controller_id}: controller evidence status is not simulated"
            )
    if controller_count == 0:
        errors.append("CFD controller coupling: missing controller contracts")
    for profile in supervisory_profile_catalog():
        profile.validate()
        if profile.evidence_status != "simulated_metadata":
            errors.append(
                f"{profile.profile_id}: supervisory evidence status is not simulated"
            )
    supervisory_record_count = 0
    for area_id in AREA_IDS:
        for record in build_reference_supervisory_records(area_id):
            record.validate()
            supervisory_record_count += 1
            if record.site_identity != "unknown":
                errors.append(f"{record.record_id}: site identity is not unknown")
            if record.evidence_status != "simulated_metadata":
                errors.append(f"{record.record_id}: evidence status is not simulated")
    if supervisory_record_count == 0:
        errors.append("CFD supervisory layer: missing reference records")
    semantic_bundle_count = 0
    for area_id in AREA_IDS:
        bundle = build_reference_operator_historian_semantics(area_id)
        bundle.validate()
        semantic_bundle_count += 1
        if bundle.evidence_status != "simulated_metadata":
            errors.append(f"{area_id}: operator/historian bundle is not simulated")
    if semantic_bundle_count == 0:
        errors.append("CFD operator/historian semantics: missing bundles")
    field_source = CalibrationEvidenceSource(
        source_id="quality-field-source",
        evidence_class=CalibrationEvidenceClass.FIELD_TELEMETRY,
        description="quality-check field telemetry placeholder",
        source_reference="quality-check:field-source",
        limitations=("quality-check source only", "not full plant validation"),
    )
    parameter = CalibrationParameter(
        parameter="quality_offset",
        value=0.1,
        units="a.u.",
        source_id=field_source.source_id,
    )
    residual = CalibrationResidual.from_values(
        parameter="quality_offset",
        observed=1.0,
        predicted=1.0,
        tolerance=0.01,
    )
    direct_assessment = assess_calibration_evidence(
        area_id="quality-area",
        sources=(field_source,),
        parameters=(parameter,),
        residuals=(residual,),
    )
    if direct_assessment.calibration_status == CalibrationStatus.EVIDENCE_CALIBRATED:
        errors.append("CFD calibration: evidence status upgraded without record")
    fit = fit_calibration_parameter(
        area_id="quality-area",
        source=field_source,
        parameter="quality_offset",
        units="a.u.",
        samples=(
            CalibrationDataPoint(
                "quality-good-1", field_source.source_id, "quality_offset", 1.1, 1.0
            ),
            CalibrationDataPoint(
                "quality-good-2", field_source.source_id, "quality_offset", 1.2, 1.1
            ),
            CalibrationDataPoint(
                "quality-rejected", "other-source", "quality_offset", 0.0, 0.0
            ),
        ),
        residual_tolerance=0.001,
        accepted_min=-1.0,
        accepted_max=1.0,
    )
    fit.validate()
    if not fit.rejected_data:
        errors.append("CFD calibration: rejected sample data was not preserved")
    fit_assessment = assess_calibration_fit(fit)
    if fit_assessment.calibration_status != CalibrationStatus.EVIDENCE_CALIBRATED:
        errors.append("CFD calibration: explicit record did not support assessment")
    suite = build_reference_numerical_verification_suite()
    suite.validate()
    if not suite.passed:
        errors.append("CFD numerical verification suite: not all cases pass")
    categories = {result.category for result in suite.results}
    required_categories = {
        "manufactured_constant_solution",
        "conservation_check",
        "mesh_refinement_sensitivity",
        "boundary_condition_check",
        "long_run_drift",
    }
    missing_categories = required_categories - categories
    if missing_categories:
        errors.append(
            "CFD numerical verification suite: missing categories "
            f"{sorted(missing_categories)}"
        )


def _check_required_test_dependencies(errors: list[str]) -> None:
    if importlib.util.find_spec("pymodbus") is None:
        errors.append(
            "pymodbus is required for the full quality gate; "
            'install with `pip install -e ".[dev,modbus]"`'
        )


def main() -> int:
    errors: list[str] = []
    files = _project_files()
    _check_folder_density(errors)
    _check_code_line_counts(files, errors)
    _check_markdown_links(files, errors)
    _check_scenario_docs(errors)
    check_hs_docs(ROOT, errors)
    _check_cfd_docs(errors)
    _check_required_test_dependencies(errors)

    if errors:
        print("HydraSim quality check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("HydraSim quality check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
