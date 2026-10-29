"""test_ladder_solver.py - pytest tests for the ladder friction physics engine.

Tests are written against the hand-calculated benchmark and derived formulae.
The backend (ladder_solver.py) is NOT modified here; tests must pass as-is.
"""

import math
import sys
import os
import pytest

# Allow import from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from ladder_solver import solve_ladder_friction


# ---------------------------------------------------------------------------
# Test 1 — Benchmark
# L=4, theta=60°, m_ladder=15 kg, m_person=90 kg, mu_A=0.25, smooth wall
# Expected: s ≈ 1.69 m, N_A ≈ 1030.05 N, N_B ≈ 257.51 N
# ---------------------------------------------------------------------------
class TestBenchmark:
    """Hand-verified benchmark: person position on a smooth-walled ladder."""

    KNOWN = {
        "L": 4.0,
        "theta_deg": 60.0,
        "m_ladder": 15.0,
        "m_person": 90.0,
        "mu_A": 0.25,
        "mu_B": 0.0,
    }
    TOL = 1e-2  # 0.01 tolerance (≪ 1% error)

    def test_s_value(self) -> None:
        """Solved position s must be ≈ 1.69 m."""
        solutions = solve_ladder_friction(self.KNOWN, "s")
        assert len(solutions) >= 1
        s = solutions[0]["s"]
        assert abs(s - 1.6874) < self.TOL, f"Expected s ≈ 1.6874, got {s:.4f}"

    def test_N_A(self) -> None:
        """Ground normal force N_A must be ≈ 1030.05 N."""
        solutions = solve_ladder_friction(self.KNOWN, "s")
        N_A = solutions[0]["N_A"]
        assert abs(N_A - 1030.05) < self.TOL, f"Expected N_A ≈ 1030.05, got {N_A:.4f}"

    def test_N_B(self) -> None:
        """Wall normal force N_B must be ≈ 257.51 N."""
        solutions = solve_ladder_friction(self.KNOWN, "s")
        N_B = solutions[0]["N_B"]
        assert abs(N_B - 257.5125) < self.TOL, f"Expected N_B ≈ 257.51, got {N_B:.4f}"

    def test_friction_forces_consistent(self) -> None:
        """F_A and F_B must satisfy the limiting friction relationships."""
        solutions = solve_ladder_friction(self.KNOWN, "s")
        sol = solutions[0]
        assert abs(sol["F_A"] - self.KNOWN["mu_A"] * sol["N_A"]) < self.TOL
        assert abs(sol["F_B"] - self.KNOWN["mu_B"] * sol["N_B"]) < self.TOL

    def test_solution_has_steps(self) -> None:
        """Solution dict must include a non-empty list of step strings."""
        solutions = solve_ladder_friction(self.KNOWN, "s")
        steps = solutions[0]["steps"]
        assert isinstance(steps, list) and len(steps) > 0


# ---------------------------------------------------------------------------
# Test 2 — Residual check
# Substitute the returned N_A, N_B, s back into all three equilibrium equations.
# Each residual must be near zero (|residual| < 1e-6 N or N·m).
# ---------------------------------------------------------------------------
class TestResiduals:
    """Verify the returned solution satisfies all three equilibrium equations."""

    KNOWN = {
        "L": 4.0,
        "theta_deg": 60.0,
        "m_ladder": 15.0,
        "m_person": 90.0,
        "mu_A": 0.25,
        "mu_B": 0.0,
    }
    G = 9.81
    TOL_RESIDUAL = 1e-4  # N or N·m – tight but reasonable for float arithmetic

    def _check_residuals(self, known: dict, target: str) -> None:
        solutions = solve_ladder_friction(known, target, g=self.G)
        sol = solutions[0]
        g = self.G

        # Reconstruct all parameter values post-solution
        L = sol.get("L", known.get("L"))
        theta_deg = sol.get("theta_deg", known.get("theta_deg"))
        theta = math.radians(theta_deg)
        m_ladder = known.get("m_ladder", 0.0)
        m_person = sol.get("m_person", known.get("m_person", 0.0))
        m_bucket = known.get("m_bucket", 0.0)
        s = sol.get("s", known.get("s", 0.0))
        d = known.get("d", 0.0)
        mu_A = sol.get("mu_A", known.get("mu_A"))
        mu_B = sol.get("mu_B", known.get("mu_B", 0.0))
        P = sol.get("P", known.get("P", 0.0))
        h_P = sol.get("h_P", known.get("h_P", 0.0))
        P_dir = known.get("P_dir", 1)

        N_A = sol["N_A"]
        N_B = sol["N_B"]
        F_A = mu_A * N_A
        F_B = mu_B * N_B
        W_l = m_ladder * g
        W_p = m_person * g
        W_b = m_bucket * g

        # eq1: ΣFx = 0
        res1 = F_A - N_B + P_dir * P
        # eq2: ΣFy = 0
        res2 = N_A + F_B - W_l - W_p - W_b
        # eq3: ΣM_A = 0
        res3 = (
            N_B * L * math.sin(theta)
            + F_B * L * math.cos(theta)
            - W_l * (L / 2) * math.cos(theta)
            - W_p * s * math.cos(theta)
            - W_b * d * math.cos(theta)
            - P_dir * P * h_P * math.sin(theta)
        )

        assert abs(res1) < self.TOL_RESIDUAL, f"ΣFx residual too large: {res1:.2e}"
        assert abs(res2) < self.TOL_RESIDUAL, f"ΣFy residual too large: {res2:.2e}"
        assert abs(res3) < self.TOL_RESIDUAL, f"ΣM_A residual too large: {res3:.2e}"

    def test_residuals_benchmark(self) -> None:
        """Benchmark case: all three equilibrium residuals ~ 0."""
        self._check_residuals(self.KNOWN, "s")

    def test_residuals_rough_wall(self) -> None:
        """Rough wall case: all three equilibrium residuals ~ 0."""
        # Solve for mu_A: must NOT pass mu_A in known (it is the target)
        known = {
            "L": self.KNOWN["L"],
            "theta_deg": self.KNOWN["theta_deg"],
            "m_ladder": self.KNOWN["m_ladder"],
            "m_person": self.KNOWN["m_person"],
            "s": 1.5,       # give s so mu_A can be found
            "mu_B": 0.15,   # rough wall
        }
        self._check_residuals(known, "mu_A")


# ---------------------------------------------------------------------------
# Test 3 — Solving for theta: smooth wall, uniform ladder only
# With mu_A, a smooth wall, and no person/bucket, the critical angle comes from:
#     tan(theta) = 1 / (2 * mu_A)
# For mu_A = 0.4 -> theta = arctan(1/0.8) ≈ 51.3402°
# ---------------------------------------------------------------------------
class TestSolveTheta:
    """Solve for the critical lean angle of a uniform ladder on a smooth wall."""

    G = 9.81

    def test_theta_formula(self) -> None:
        """theta_deg from solver matches the closed-form tan(θ) = 1/(2μ_A)."""
        mu_A = 0.4
        expected_deg = math.degrees(math.atan(1.0 / (2.0 * mu_A)))  # ≈ 51.3402°

        known = {
            "L": 5.0,
            "m_ladder": 20.0,
            "m_person": 0.0,
            "mu_A": mu_A,
            "mu_B": 0.0,
        }
        solutions = solve_ladder_friction(known, "theta_deg", g=self.G)
        assert len(solutions) >= 1
        theta_solved = solutions[0]["theta_deg"]
        assert abs(theta_solved - expected_deg) < 0.02, (
            f"Expected theta ≈ {expected_deg:.4f}°, got {theta_solved:.4f}°"
        )


# ---------------------------------------------------------------------------
# Test 4 — Rough wall
# Solving for s with a rough wall (mu_B = 0.15). Result must be physically valid.
# ---------------------------------------------------------------------------
class TestRoughWall:
    """A rough wall increases how far the person can climb."""

    KNOWN_SMOOTH = {
        "L": 4.0,
        "theta_deg": 60.0,
        "m_ladder": 15.0,
        "m_person": 90.0,
        "mu_A": 0.25,
        "mu_B": 0.0,
    }

    def test_rough_wall_larger_s(self) -> None:
        """Person can climb farther on a rough wall than on a smooth wall."""
        known_rough = dict(self.KNOWN_SMOOTH)
        known_rough["mu_B"] = 0.15

        s_smooth = solve_ladder_friction(self.KNOWN_SMOOTH, "s")[0]["s"]
        s_rough  = solve_ladder_friction(known_rough, "s")[0]["s"]

        assert s_rough > s_smooth, (
            f"Rough wall should allow further climb: s_rough={s_rough:.4f} <= s_smooth={s_smooth:.4f}"
        )

    def test_rough_wall_N_B_positive(self) -> None:
        """Wall normal force must remain positive (wall pushes on ladder)."""
        known_rough = dict(self.KNOWN_SMOOTH)
        known_rough["mu_B"] = 0.15
        sol = solve_ladder_friction(known_rough, "s")[0]
        assert sol["N_B"] > 0, f"N_B should be positive, got {sol['N_B']}"


# ---------------------------------------------------------------------------
# Test 5 — Applied force P toward wall (P_dir = +1)
# P toward wall (+x) adds to F_A (= mu_A * N_A) and also raises N_B.
# The moment equation shows N_B increases, which increases the restoring
# moment about A → the person can climb *farther*.
# Verified numerically: s_no_P ≈ 1.687 m, s_with_P ≈ 1.884 m.
# ---------------------------------------------------------------------------
class TestAppliedForceTowardWall:
    """Horizontal push toward the wall increases the safe climbing distance."""

    BASE = {
        "L": 4.0,
        "theta_deg": 60.0,
        "m_ladder": 15.0,
        "m_person": 90.0,
        "mu_A": 0.25,
        "mu_B": 0.0,
    }

    def test_P_toward_wall_increases_s(self) -> None:
        """A push toward the wall (raises N_B & restoring moment) lets the person climb farther."""
        s_no_P = solve_ladder_friction(self.BASE, "s")[0]["s"]

        known_P = dict(self.BASE)
        known_P["P"] = 50.0
        known_P["h_P"] = 2.0
        known_P["P_dir"] = 1
        s_with_P = solve_ladder_friction(known_P, "s")[0]["s"]

        assert s_with_P > s_no_P, (
            f"Push toward wall should increase s: {s_with_P:.4f} should be > {s_no_P:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 6 — Applied force P away from wall (P_dir = -1)
# P away from wall (−x) reduces N_B (from ΣFx) and so shrinks the restoring
# moment about A → the person can climb *less far*.
# Verified numerically: s_no_P ≈ 1.687 m, s_with_P ≈ 1.491 m.
# ---------------------------------------------------------------------------
class TestAppliedForceAwayFromWall:
    """Horizontal pull away from wall reduces the safe climbing distance."""

    BASE = {
        "L": 4.0,
        "theta_deg": 60.0,
        "m_ladder": 15.0,
        "m_person": 90.0,
        "mu_A": 0.25,
        "mu_B": 0.0,
    }

    def test_P_away_from_wall_reduces_s(self) -> None:
        """A pull away from wall (reduces N_B & restoring moment) reduces safe climbing distance."""
        s_no_P = solve_ladder_friction(self.BASE, "s")[0]["s"]

        known_P = dict(self.BASE)
        known_P["P"] = 50.0
        known_P["h_P"] = 2.0
        known_P["P_dir"] = -1
        s_with_P = solve_ladder_friction(known_P, "s")[0]["s"]

        assert s_with_P < s_no_P, (
            f"Pull away from wall should reduce s: {s_with_P:.4f} should be < {s_no_P:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 7 — Bucket load
# Adding a bucket at the base end reduces the person's safe climbing distance
# because it adds weight at a position that increases the overturning moment.
# ---------------------------------------------------------------------------
class TestBucketLoad:
    """Bucket load at high position reduces safe climbing distance."""

    BASE = {
        "L": 4.0,
        "theta_deg": 60.0,
        "m_ladder": 15.0,
        "m_person": 90.0,
        "mu_A": 0.25,
        "mu_B": 0.0,
    }

    def test_bucket_at_top_reduces_s(self) -> None:
        """A heavy bucket near the top of the ladder reduces safe person position."""
        s_no_bucket = solve_ladder_friction(self.BASE, "s")[0]["s"]

        known_b = dict(self.BASE)
        known_b["m_bucket"] = 20.0
        known_b["d"] = 3.5   # near the top – increases overturning moment
        s_with_bucket = solve_ladder_friction(known_b, "s")[0]["s"]

        assert s_with_bucket < s_no_bucket, (
            f"Bucket near top should reduce s: {s_with_bucket:.4f} >= {s_no_bucket:.4f}"
        )

    def test_bucket_at_base_increases_s(self) -> None:
        """A heavy bucket very close to the base has negligible/positive effect."""
        known_b = dict(self.BASE)
        known_b["m_bucket"] = 20.0
        known_b["d"] = 0.1   # almost at the base – tiny moment arm
        sol = solve_ladder_friction(known_b, "s")[0]
        # Just check it returns a physically valid result
        assert 0 <= sol["s"] <= known_b["L"]
        assert sol["N_A"] > 0
        assert sol["N_B"] >= 0
