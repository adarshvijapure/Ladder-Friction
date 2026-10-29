"""verify_smoke.py - Manual smoke test for the solver API (simulates UI calls)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ladder_solver import solve_ladder_friction

# --- Test 1: Ques 6, theta=60 ---
known = {
    "L": 4.0,
    "theta_deg": 60.0,
    "m_ladder": 15.0,
    "m_person": 90.0,
    "mu_A": 0.25,
    "mu_B": 0.0,
    "m_bucket": 0.0,
    "d": 0.0,
    "P": 0.0,
    "h_P": 0.0,
    "P_dir": 1,
}
sols = solve_ladder_friction(known, "s")
sol = sols[0]
print("=== TEST 1: Ques 6 (theta=60, solve s) ===")
print(f"s    = {sol['s']:.4f} m  (expected ~1.69)")
print(f"N_A  = {sol['N_A']:.2f} N  (expected ~1030.05)")
print(f"N_B  = {sol['N_B']:.2f} N  (expected ~257.51)")
print(f"F_A  = {sol['F_A']:.2f} N")
print(f"F_B  = {sol['F_B']:.2f} N")
print(f"Steps count: {len(sol['steps'])}")
print()

# --- Test 2: Error – mu_A is target but also in known ---
print("=== TEST 2: Error – target also in known ===")
known2 = dict(known)
known2["mu_A"] = 0.25
try:
    solve_ladder_friction(known2, "mu_A")
    print("FAIL: no error raised!")
except ValueError as e:
    print(f"OK – ValueError: {e}")
print()

# --- Test 3: Error – theta_deg missing ---
print("=== TEST 3: Error – theta_deg missing ===")
known3 = {k: v for k, v in known.items() if k != "theta_deg"}
try:
    solve_ladder_friction(known3, "s")
    print("FAIL: no error raised!")
except ValueError as e:
    print(f"OK – ValueError: {e}")
