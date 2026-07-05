"""Profile registry for the reference water plant."""

from __future__ import annotations

from .models import PlantProfile
from .profile_data import PROFILES
from .validator import assert_valid_profile


def profile_ids() -> tuple[str, ...]:
    return tuple(profile.profile_id for profile in PROFILES)


def get_profile(profile_id: str) -> PlantProfile:
    for profile in PROFILES:
        if profile.profile_id == profile_id:
            return assert_valid_profile(profile)
    known = ", ".join(profile_ids())
    raise ValueError(f"unknown plant profile {profile_id!r}; expected one of: {known}")
