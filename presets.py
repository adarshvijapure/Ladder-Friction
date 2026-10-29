"""presets.py - Preset problem definitions for the Ladder Friction Solver UI.

Each preset is a dict matching the `known` parameter of solve_ladder_friction(),
plus a 'target' key indicating the variable to solve for.
A value of None means the UI must prompt the user for that input before solving.
"""

from typing import Dict, Any, Optional


# Each preset dict must contain:
#   'label'     : str           - Display name shown in the selectbox
#   'target'    : str           - Variable to solve for (must be a VALID_TARGET)
#   'known'     : dict          - Parameters passed to solve_ladder_friction();
#                                 None values signal a required user input
#   'notes'     : str           - Short description displayed in the UI

PRESETS: Dict[str, Dict[str, Any]] = {
    "Ques 6": {
        "label": "Ques 6",
        "target": "s",
        "notes": (
            "A 15 kg, 4 m ladder leans against a smooth wall. "
            "A 90 kg person climbs up. mu_A = 0.25. "
            "Find how far the person can climb before slipping (angle to be supplied)."
        ),
        "known": {
            "L": 4.0,
            "m_ladder": 15.0,
            "m_person": 90.0,
            "mu_A": 0.25,
            "mu_B": 0.0,
            # theta_deg is intentionally None – UI must ask before enabling Solve
            "theta_deg": None,  # TODO: to be supplied by user
            # Optional loads not in this problem
            "m_bucket": 0.0,
            "d": 0.0,
            "P": 0.0,
            "h_P": 0.0,
            "P_dir": 1,
        },
    },

    "Custom": {
        "label": "Custom",
        "target": "s",
        "notes": "Enter your own values. All fields are editable.",
        "known": {
            "L": 5.0,
            "theta_deg": 60.0,
            "m_ladder": 20.0,
            "m_person": 80.0,
            "s": None,          # will be disabled (it's the target)
            "m_bucket": 0.0,
            "d": 0.0,
            "mu_A": 0.3,
            "mu_B": 0.0,
            "P": 0.0,
            "h_P": 0.0,
            "P_dir": 1,
        },
    },
}

# Ordered list of preset names for the selectbox
PRESET_NAMES = list(PRESETS.keys())
