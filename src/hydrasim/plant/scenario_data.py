"""Built-in staged water-plant scenarios."""

from __future__ import annotations

from .models import HsScenario
from .scenario_faults import FAULT_SCENARIOS
from .scenario_review import REVIEW_SCENARIOS
from .scenario_startup import STARTUP_SCENARIOS

SCENARIOS: tuple[HsScenario, ...] = (
    STARTUP_SCENARIOS + FAULT_SCENARIOS + REVIEW_SCENARIOS
)
