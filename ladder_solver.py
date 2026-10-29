"""ladder_solver.py - Pure-Python physics engine for ladder friction.

Conventions & Equilibrium:
- Origin A at ladder base (0, 0). Wall on the right at x = L*cos(theta).
- B is top of ladder at (L*cos(theta), L*sin(theta)).
- Forces:
    N_A: Ground normal force, acts +y (upwards).
    F_A = mu_A * N_A: Ground friction force, acts +x (toward wall, limiting friction).
    N_B: Wall normal force, acts -x (pushing left away from wall).
    F_B = mu_B * N_B: Wall friction force, acts +y (upwards, limiting friction).
    W_ladder = m_ladder * g: Ladder weight at L/2.
    W_person = m_person * g: Person weight at distance s along ladder.
    W_bucket = m_bucket * g: Bucket weight at distance d along ladder.
    P: Optional horizontal force at distance h_P along ladder.
       P_dir = +1 (toward wall, +x) or -1 (away from wall, -x).
- Equilibrium Equations:
    Sum F_x = 0:  F_A - N_B + P_dir * P = 0
    Sum F_y = 0:  N_A + F_B - W_ladder - W_person - W_bucket = 0
    Sum M_A = 0 (CCW positive about A):
        N_B * L * sin(theta) + F_B * L * cos(theta)
        - W_ladder * (L / 2) * cos(theta)
        - W_person * s * cos(theta)
        - W_bucket * d * cos(theta)
        - P_dir * P * h_P * sin(theta) = 0
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
import sympy as sp

VALID_TARGETS = {
    "s",
    "theta_deg",
    "mu_A",
    "mu_B",
    "P",
    "h_P",
    "m_person",
    "L",
}


def _validate_inputs(known: Dict[str, Any], target: str) -> Dict[str, Any]:
    """Validate input parameters and return a sanitized dictionary with defaults."""
    if target not in VALID_TARGETS:
        raise ValueError(
            f"Invalid target '{target}'. Allowed targets are: {sorted(list(VALID_TARGETS))}"
        )

    # Check that target is not given a concrete value in known
    if target in known and known[target] is not None:
        raise ValueError(
            f"Target variable '{target}' cannot be given as a known value in 'known'. "
            "Remove it or set it to None."
        )

    clean: Dict[str, Any] = {}

    # Defaults for optional loads/forces
    defaults = {
        "m_person": 0.0,
        "s": 0.0,
        "m_bucket": 0.0,
        "d": 0.0,
        "mu_B": 0.0,
        "P": 0.0,
        "h_P": 0.0,
        "P_dir": 1,
    }

    # Populate known values and defaults
    for k, default_val in defaults.items():
        if k == target:
            continue
        val = known.get(k, default_val)
        clean[k] = default_val if val is None else val

    # Non-optional parameters (must be present if they are not the target)
    core_keys = ["L", "m_ladder", "mu_A", "theta_deg"]
    missing = []
    for k in core_keys:
        if k == target:
            continue
        if k not in known or known[k] is None:
            missing.append(k)
        else:
            clean[k] = known[k]

    if missing:
        raise ValueError(
            f"Missing required parameter(s): {missing}. Exactly one unknown ('{target}') is allowed."
        )

    # Validate physical positivity for non-target inputs
    if "L" in clean and clean["L"] <= 0:
        raise ValueError(f"Ladder length L must be strictly positive (got {clean['L']}).")
    if "m_ladder" in clean and clean["m_ladder"] < 0:
        raise ValueError(f"Ladder mass m_ladder cannot be negative (got {clean['m_ladder']}).")
    if "mu_A" in clean and clean["mu_A"] < 0:
        raise ValueError(f"Friction coefficient mu_A cannot be negative (got {clean['mu_A']}).")
    if "mu_B" in clean and clean["mu_B"] < 0:
        raise ValueError(f"Friction coefficient mu_B cannot be negative (got {clean['mu_B']}).")
    if "m_person" in clean and clean["m_person"] < 0:
        raise ValueError(f"Person mass m_person cannot be negative (got {clean['m_person']}).")
    if "m_bucket" in clean and clean["m_bucket"] < 0:
        raise ValueError(f"Bucket mass m_bucket cannot be negative (got {clean['m_bucket']}).")
    if "P" in clean and clean["P"] < 0:
        raise ValueError(f"Force magnitude P cannot be negative (got {clean['P']}). Use P_dir for direction.")
    if "theta_deg" in clean:
        th = clean["theta_deg"]
        if th <= 0 or th >= 90:
            raise ValueError(
                f"Ladder angle theta_deg must be in (0, 90) degrees to lean against wall (got {th})."
            )

    # Validate direction
    p_dir = clean.get("P_dir", 1)
    if p_dir not in (1, -1, 1.0, -1.0):
        raise ValueError(f"P_dir must be +1 (toward wall) or -1 (away from wall), got {p_dir}.")
    clean["P_dir"] = int(p_dir)

    # Validate target-specific solvable conditions
    if target == "s" and clean["m_person"] <= 0:
        raise ValueError("Cannot solve for 's' because person mass m_person is 0 kg.")
    if target == "h_P" and clean["P"] == 0:
        raise ValueError("Cannot solve for 'h_P' because applied horizontal force P is 0 N.")

    # Location bounds for known load positions relative to L
    if "L" in clean:
        L_val = clean["L"]
        for pos_key in ("s", "d", "h_P"):
            if pos_key in clean and pos_key != target:
                val = clean[pos_key]
                if val < 0 or val > L_val + 1e-6:
                    raise ValueError(
                        f"Position '{pos_key}' ({val}) is outside ladder bounds [0, L={L_val}]."
                    )

    return clean


def _generate_step_strings(
    sol_dict: Dict[str, Any],
    params: Dict[str, Any],
    target: str,
    g: float,
) -> List[str]:
    """Generate human-readable step-by-step equilibrium derivation."""
    steps = []
    L = params["L"]
    theta_deg = params["theta_deg"]
    theta_rad = math.radians(theta_deg)
    m_ladder = params["m_ladder"]
    m_person = params["m_person"]
    m_bucket = params["m_bucket"]
    s = params["s"]
    d = params["d"]
    mu_A = params["mu_A"]
    mu_B = params["mu_B"]
    P = params["P"]
    h_P = params["h_P"]
    P_dir = params["P_dir"]

    N_A = sol_dict["N_A"]
    N_B = sol_dict["N_B"]
    F_A = sol_dict["F_A"]
    F_B = sol_dict["F_B"]
    target_val = sol_dict[target]

    W_l = m_ladder * g
    W_p = m_person * g
    W_b = m_bucket * g
    W_total = W_l + W_p + W_b

    steps.append(
        f"1. Gravitational Weights (g = {g:.2f} m/s²):\n"
        f"   - Ladder Weight: W_ladder = m_ladder * g = {m_ladder:.2f} kg * {g:.2f} m/s² = {W_l:.2f} N (at L/2 = {L/2:.2f} m)\n"
        f"   - Person Weight: W_person = m_person * g = {m_person:.2f} kg * {g:.2f} m/s² = {W_p:.2f} N (at s = {s:.3f} m)\n"
        f"   - Bucket Weight: W_bucket = m_bucket * g = {m_bucket:.2f} kg * {g:.2f} m/s² = {W_b:.2f} N (at d = {d:.3f} m)\n"
        f"   - Total Downward Weight: W_total = {W_total:.2f} N"
    )

    p_dir_desc = "toward wall (+x)" if P_dir > 0 else "away from wall (-x)"
    steps.append(
        f"2. Horizontal Equilibrium (ΣFx = 0):\n"
        f"   Equation: F_A - N_B + (P_dir * P) = 0\n"
        f"   Limiting ground friction: F_A = mu_A * N_A\n"
        f"   Substitution: {mu_A:.4f} * N_A - N_B + ({P_dir:+d} * {P:.2f} N) = 0\n"
        f"   (Applied force P = {P:.2f} N {p_dir_desc})"
    )

    steps.append(
        f"3. Vertical Equilibrium (ΣFy = 0):\n"
        f"   Equation: N_A + F_B - W_ladder - W_person - W_bucket = 0\n"
        f"   Limiting wall friction: F_B = mu_B * N_B\n"
        f"   Substitution: N_A + ({mu_B:.4f} * N_B) - {W_total:.2f} N = 0"
    )

    steps.append(
        f"4. Rotational Equilibrium about Base A (ΣM_A = 0, Counterclockwise Positive):\n"
        f"   Equation:\n"
        f"     + N_B * L * sin(θ) + F_B * L * cos(θ)\n"
        f"     - W_ladder * (L/2) * cos(θ) - W_person * s * cos(θ) - W_bucket * d * cos(θ)\n"
        f"     - P_dir * P * h_P * sin(θ) = 0\n"
        f"   At θ = {theta_deg:.2f}° (sin(θ) = {math.sin(theta_rad):.4f}, cos(θ) = {math.cos(theta_rad):.4f}):\n"
        f"     + N_B * {L:.2f} * {math.sin(theta_rad):.4f} + ({mu_B:.4f}*N_B) * {L:.2f} * {math.cos(theta_rad):.4f}\n"
        f"     - ({W_l:.2f} * {L/2:.2f} * {math.cos(theta_rad):.4f}) - ({W_p:.2f} * {s:.3f} * {math.cos(theta_rad):.4f})\n"
        f"     - ({W_b:.2f} * {d:.3f} * {math.cos(theta_rad):.4f}) - ({P_dir:+d} * {P:.2f} * {h_P:.3f} * {math.sin(theta_rad):.4f}) = 0"
    )

    steps.append(
        f"5. Solved Contact Forces & Target Unknown:\n"
        f"   - Target Unknown: {target} = {target_val:.4f}\n"
        f"   - Ground Normal Force: N_A = {N_A:.2f} N\n"
        f"   - Wall Normal Force: N_B = {N_B:.2f} N\n"
        f"   - Ground Limiting Friction: F_A = {F_A:.2f} N\n"
        f"   - Wall Limiting Friction: F_B = {F_B:.2f} N"
    )

    return steps


def solve_ladder_friction(known: dict, target: str, g: float = 9.81) -> list[dict]:
    """Solve the 2D ladder friction problem at impending slip.

    Args:
        known: dict of known problem parameters. Supported keys:
            L, theta_deg, m_ladder, m_person, s, m_bucket, d, mu_A, mu_B, P, h_P, P_dir
        target: name of parameter to solve for. Supported:
            's', 'theta_deg', 'mu_A', 'mu_B', 'P', 'h_P', 'm_person', 'L'
        g: acceleration due to gravity (default: 9.81 m/s^2)

    Returns:
        List of solution dictionaries containing:
            {target: float, 'N_A': float, 'N_B': float, 'F_A': float, 'F_B': float,
             'steps': list[str], 'target': target}

    Raises:
        ValueError: if inputs are invalid, under/over-determined, or physically impossible.
    """
    clean_params = _validate_inputs(known, target)

    # Setup SymPy symbols
    N_A_sym = sp.Symbol("N_A", real=True)
    N_B_sym = sp.Symbol("N_B", real=True)

    if target == "theta_deg":
        target_sym = sp.Symbol("theta_rad", real=True)
    else:
        target_sym = sp.Symbol(target, real=True)

    # Build dictionary of symbols or values for each variable
    var_map: Dict[str, Any] = {}
    for k, v in clean_params.items():
        if k == "theta_deg":
            var_map["theta"] = math.radians(v)
        else:
            var_map[k] = v

    if target == "theta_deg":
        var_map["theta"] = target_sym
    else:
        var_map[target] = target_sym

    # Quantities
    L = var_map["L"]
    theta = var_map["theta"]
    m_ladder = var_map["m_ladder"]
    m_person = var_map["m_person"]
    m_bucket = var_map["m_bucket"]
    s = var_map["s"]
    d = var_map["d"]
    mu_A = var_map["mu_A"]
    mu_B = var_map["mu_B"]
    P = var_map["P"]
    h_P = var_map["h_P"]
    P_dir = var_map["P_dir"]

    F_A = mu_A * N_A_sym
    F_B = mu_B * N_B_sym
    W_ladder = m_ladder * g
    W_person = m_person * g
    W_bucket = m_bucket * g

    # Equations of equilibrium
    # 1. Sum Fx = 0
    eq1 = F_A - N_B_sym + P_dir * P
    # 2. Sum Fy = 0
    eq2 = N_A_sym + F_B - W_ladder - W_person - W_bucket
    # 3. Sum M_A = 0
    eq3 = (
        N_B_sym * L * sp.sin(theta)
        + F_B * L * sp.cos(theta)
        - W_ladder * (L / 2) * sp.cos(theta)
        - W_person * s * sp.cos(theta)
        - W_bucket * d * sp.cos(theta)
        - P_dir * P * h_P * sp.sin(theta)
    )

    # Solve the system
    raw_solutions = sp.solve([eq1, eq2, eq3], [target_sym, N_A_sym, N_B_sym], dict=True)

    if not raw_solutions:
        raise ValueError(
            f"No mathematical solution found for target '{target}' with given parameters."
        )

    valid_solutions: List[dict] = []
    rejected_reasons: List[str] = []

    seen_signatures = set()

    for sol in raw_solutions:
        # Extract values
        if target_sym not in sol or N_A_sym not in sol or N_B_sym not in sol:
            continue

        raw_target = sol[target_sym]
        raw_NA = sol[N_A_sym]
        raw_NB = sol[N_B_sym]

        try:
            val_target = complex(raw_target.evalf())
            val_NA = complex(raw_NA.evalf())
            val_NB = complex(raw_NB.evalf())
        except Exception:
            continue

        # Drop non-real solutions
        if (
            abs(val_target.imag) > 1e-7
            or abs(val_NA.imag) > 1e-7
            or abs(val_NB.imag) > 1e-7
        ):
            rejected_reasons.append("Complex/imaginary root dropped")
            continue

        target_real = float(val_target.real)
        na_real = float(val_NA.real)
        nb_real = float(val_NB.real)

        # Numerical cleanup for tiny near-zero values
        if abs(na_real) < 1e-7:
            na_real = 0.0
        if abs(nb_real) < 1e-7:
            nb_real = 0.0

        # Physical validation for normal forces
        if na_real < -1e-6:
            rejected_reasons.append(
                f"Negative ground normal force N_A = {na_real:.2f} N (ladder lifts off ground)"
            )
            continue
        if nb_real < -1e-6:
            rejected_reasons.append(
                f"Negative wall normal force N_B = {nb_real:.2f} N (ladder pulls away from wall)"
            )
            continue

        # Check target physical admissibility
        admissible = True
        reason = ""

        if target == "theta_deg":
            # target_real is in radians internally
            theta_deg_val = math.degrees(target_real)
            # Normalize periodic angle to (-180, 180]
            theta_deg_val = (theta_deg_val + 180) % 360 - 180
            if theta_deg_val <= 1e-5 or theta_deg_val >= 90.0 - 1e-5:
                admissible = False
                reason = f"Ladder angle theta = {theta_deg_val:.2f}° outside physical range (0°, 90°)"
            final_target_val = theta_deg_val
        else:
            final_target_val = target_real

        if target == "s":
            L_val = clean_params["L"]
            if final_target_val < -1e-5 or final_target_val > L_val + 1e-5:
                admissible = False
                reason = f"Person position s = {final_target_val:.3f} m outside ladder length [0, {L_val:.3f} m]"
        elif target == "h_P":
            L_val = clean_params["L"]
            if final_target_val < -1e-5 or final_target_val > L_val + 1e-5:
                admissible = False
                reason = f"Force position h_P = {final_target_val:.3f} m outside ladder length [0, {L_val:.3f} m]"
        elif target in ("mu_A", "mu_B"):
            if final_target_val < -1e-6:
                admissible = False
                reason = f"Friction coefficient {target} = {final_target_val:.4f} cannot be negative"
        elif target == "m_person":
            if final_target_val < -1e-6:
                admissible = False
                reason = f"Person mass m_person = {final_target_val:.2f} kg cannot be negative"
        elif target == "L":
            max_pos = max(
                clean_params.get("s", 0.0),
                clean_params.get("d", 0.0),
                clean_params.get("h_P", 0.0),
            )
            if final_target_val <= 0:
                admissible = False
                reason = f"Ladder length L = {final_target_val:.2f} m must be strictly positive"
            elif final_target_val < max_pos - 1e-5:
                admissible = False
                reason = f"Ladder length L = {final_target_val:.2f} m cannot be shorter than load position ({max_pos:.2f} m)"
        elif target == "P":
            if final_target_val < -1e-6:
                admissible = False
                reason = f"Force magnitude P = {final_target_val:.2f} N cannot be negative"

        if not admissible:
            rejected_reasons.append(reason)
            continue

        # Prevent duplicate solutions
        sig = (round(final_target_val, 5), round(na_real, 5), round(nb_real, 5))
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        # Compute friction forces
        mu_A_val = final_target_val if target == "mu_A" else clean_params["mu_A"]
        mu_B_val = final_target_val if target == "mu_B" else clean_params["mu_B"]
        F_A_val = float(mu_A_val * na_real)
        F_B_val = float(mu_B_val * nb_real)

        # Prepare full parameter dictionary for steps
        eval_params = dict(clean_params)
        eval_params[target] = final_target_val

        sol_dict = {
            target: float(final_target_val),
            "N_A": float(na_real),
            "N_B": float(nb_real),
            "F_A": float(F_A_val),
            "F_B": float(F_B_val),
            "target": target,
        }

        sol_dict["steps"] = _generate_step_strings(sol_dict, eval_params, target, g)
        valid_solutions.append(sol_dict)

    if not valid_solutions:
        reasons_text = "; ".join(rejected_reasons) if rejected_reasons else "equations yield no valid real roots"
        raise ValueError(
            f"No physically admissible solution found for target '{target}'. Rejected due to: {reasons_text}"
        )

    return valid_solutions
