"""app.py - Streamlit frontend for the Ladder Friction Solver.

Calls ladder_solver.solve_ladder_friction(); contains NO physics logic.
Layout: sidebar (preset + target) | left col (inputs) | right col (diagram).
"""

import math
import sys
import os

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Allow import from same directory regardless of working dir
sys.path.insert(0, os.path.dirname(__file__))

from ladder_solver import solve_ladder_friction, VALID_TARGETS
from presets import PRESETS, PRESET_NAMES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ladder Friction Solver",
    page_icon="🪜",
    layout="wide",
)
st.title("🪜 Ladder Friction Solver")
st.caption("First-Year Engineering Mechanics · 2D Equilibrium at Impending Slip")

# ---------------------------------------------------------------------------
# Sidebar: preset selector + target selector
# ---------------------------------------------------------------------------
st.sidebar.header("Problem Setup")

selected_preset_name: str = st.sidebar.selectbox(
    "Preset Problem",
    options=PRESET_NAMES,
    index=0,
    key="preset_select",
)

# Human-readable target labels
TARGET_LABELS: dict[str, str] = {
    "s":         "s — Person position along ladder (m)",
    "theta_deg": "θ — Ladder angle with horizontal (°)",
    "mu_A":      "μ_A — Ground friction coefficient",
    "mu_B":      "μ_B — Wall friction coefficient",
    "P":         "P — Applied horizontal force (N)",
    "h_P":       "h_P — Force application point along ladder (m)",
    "m_person":  "m_person — Person mass (kg)",
    "L":         "L — Ladder length (m)",
}

preset_data = PRESETS[selected_preset_name]
preset_target: str = preset_data["target"]
preset_known: dict = preset_data["known"]

# Show preset notes
if preset_data.get("notes"):
    st.sidebar.info(preset_data["notes"])

# Allow user to override the target
target: str = st.sidebar.selectbox(
    "Solve For",
    options=list(VALID_TARGETS),
    index=list(VALID_TARGETS).index(preset_target),
    format_func=lambda t: TARGET_LABELS.get(t, t),
    key="target_select",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Gravity**")
g_val: float = st.sidebar.number_input(
    "g (m/s²)", value=9.81, min_value=1.0, max_value=20.0, step=0.01, key="g_input"
)

# ---------------------------------------------------------------------------
# Helper: load a preset value with fallback
# ---------------------------------------------------------------------------
def preset_val(key: str, fallback: float) -> float:
    """Return numeric value from preset, substituting fallback for None."""
    v = preset_known.get(key)
    return fallback if v is None else float(v)


# ---------------------------------------------------------------------------
# Left column: all parameter inputs
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Input Parameters")
    st.caption("The field being solved for is disabled (greyed out).")

    # ---- Ladder geometry ----
    st.markdown("**Ladder Geometry**")

    L_disabled = (target == "L")
    L_val: float = st.number_input(
        "L — Ladder length (m)",
        value=preset_val("L", 5.0),
        min_value=0.1,
        max_value=30.0,
        step=0.1,
        disabled=L_disabled,
        key="L_input",
    )

    theta_disabled = (target == "theta_deg")
    # Theta may be None in preset (user must supply it)
    theta_preset_raw = preset_known.get("theta_deg")
    theta_needs_user_input: bool = (theta_preset_raw is None) and (target != "theta_deg")
    theta_default: float = 60.0 if theta_preset_raw is None else float(theta_preset_raw)

    theta_val: float = st.number_input(
        "θ — Ladder angle with horizontal (°)"
        + (" ⚠ Required for this preset" if theta_needs_user_input else ""),
        value=theta_default,
        min_value=1.0,
        max_value=89.0,
        step=0.5,
        disabled=theta_disabled,
        key="theta_input",
    )

    # ---- Masses ----
    st.markdown("**Masses**")

    m_ladder_val: float = st.number_input(
        "m_ladder — Ladder mass (kg)",
        value=preset_val("m_ladder", 20.0),
        min_value=0.0,
        max_value=500.0,
        step=0.5,
        key="m_ladder_input",
    )

    m_person_disabled = (target == "m_person")
    m_person_val: float = st.number_input(
        "m_person — Person mass (kg)",
        value=preset_val("m_person", 80.0),
        min_value=0.0,
        max_value=500.0,
        step=0.5,
        disabled=m_person_disabled,
        key="m_person_input",
    )

    m_bucket_val: float = st.number_input(
        "m_bucket — Bucket mass (kg)",
        value=preset_val("m_bucket", 0.0),
        min_value=0.0,
        max_value=200.0,
        step=0.5,
        key="m_bucket_input",
    )

    # ---- Load positions ----
    st.markdown("**Load Positions (along ladder from base A)**")

    s_disabled = (target == "s")
    s_val: float = st.number_input(
        "s — Person position (m)",
        value=preset_val("s", 0.0),
        min_value=0.0,
        max_value=30.0,
        step=0.05,
        disabled=s_disabled,
        key="s_input",
    )

    d_val: float = st.number_input(
        "d — Bucket position (m)",
        value=preset_val("d", 0.0),
        min_value=0.0,
        max_value=30.0,
        step=0.05,
        key="d_input",
    )

    # ---- Friction ----
    st.markdown("**Friction Coefficients**")

    mu_A_disabled = (target == "mu_A")
    mu_A_val: float = st.number_input(
        "μ_A — Ground (static)",
        value=preset_val("mu_A", 0.3),
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        disabled=mu_A_disabled,
        key="mu_A_input",
    )

    mu_B_disabled = (target == "mu_B")
    mu_B_val: float = st.number_input(
        "μ_B — Wall (static, 0 = smooth)",
        value=preset_val("mu_B", 0.0),
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        disabled=mu_B_disabled,
        key="mu_B_input",
    )

    # ---- Applied horizontal force ----
    st.markdown("**Applied Horizontal Force (optional)**")

    P_disabled = (target == "P")
    P_val: float = st.number_input(
        "P — Force magnitude (N)",
        value=preset_val("P", 0.0),
        min_value=0.0,
        max_value=10000.0,
        step=1.0,
        disabled=P_disabled,
        key="P_input",
    )

    h_P_disabled = (target == "h_P")
    h_P_val: float = st.number_input(
        "h_P — Application point along ladder (m)",
        value=preset_val("h_P", 0.0),
        min_value=0.0,
        max_value=30.0,
        step=0.05,
        disabled=h_P_disabled,
        key="h_P_input",
    )

    P_dir_val: int = st.radio(
        "P direction",
        options=[1, -1],
        format_func=lambda x: "→ Toward wall (+x)" if x == 1 else "← Away from wall (−x)",
        index=0 if preset_val("P_dir", 1) >= 0 else 1,
        horizontal=True,
        key="P_dir_input",
    )

    st.markdown("---")

    # ---- Solve button ----
    # Disable if theta is required but still at the default placeholder
    # (we check: preset has theta_deg=None AND target != theta_deg AND user hasn't changed it)
    # We can't detect "not yet typed" reliably – instead gate on the value being valid (>0 & <90)
    theta_ok = (target == "theta_deg") or (1.0 <= theta_val <= 89.0)
    solve_blocked: bool = theta_needs_user_input and not theta_ok

    if solve_blocked:
        st.warning("⚠ Please enter a valid angle θ (1°–89°) for this preset before solving.")

    solve_clicked: bool = st.button(
        "⚡ Solve",
        type="primary",
        disabled=solve_blocked,
        use_container_width=True,
        key="solve_btn",
    )

# ---------------------------------------------------------------------------
# Build the `known` dict for the solver from current UI values
# ---------------------------------------------------------------------------
known_ui: dict = {
    "L":        L_val       if target != "L"         else None,
    "theta_deg": theta_val  if target != "theta_deg" else None,
    "m_ladder": m_ladder_val,
    "m_person": m_person_val if target != "m_person"  else None,
    "s":        s_val       if target != "s"          else None,
    "m_bucket": m_bucket_val,
    "d":        d_val,
    "mu_A":     mu_A_val    if target != "mu_A"       else None,
    "mu_B":     mu_B_val    if target != "mu_B"       else None,
    "P":        P_val       if target != "P"          else None,
    "h_P":      h_P_val     if target != "h_P"        else None,
    "P_dir":    P_dir_val,
}

# ---------------------------------------------------------------------------
# Session state: keep last solution across re-runs
# ---------------------------------------------------------------------------
if "last_solution" not in st.session_state:
    st.session_state["last_solution"] = None

if solve_clicked:
    try:
        solutions = solve_ladder_friction(known_ui, target, g=g_val)
        st.session_state["last_solution"] = {
            "sol": solutions[0],          # take first (usually only) valid solution
            "known_snapshot": dict(known_ui),
            "target": target,
        }
        if len(solutions) > 1:
            st.info(f"Multiple physically admissible solutions found; showing the first of {len(solutions)}.")
    except ValueError as exc:
        st.session_state["last_solution"] = None
        st.error(f"**Physics Error:** {exc}")

# ---------------------------------------------------------------------------
# Results area (below the two-column layout)
# ---------------------------------------------------------------------------
last = st.session_state.get("last_solution")

if last is not None:
    sol = last["sol"]
    tgt = last["target"]

    st.divider()
    st.subheader("Results")

    # Solved target metric
    tgt_label = TARGET_LABELS.get(tgt, tgt)
    tgt_val_display = sol[tgt]
    unit_map = {
        "s": "m", "theta_deg": "°", "mu_A": "", "mu_B": "",
        "P": "N", "h_P": "m", "m_person": "kg", "L": "m",
    }
    unit = unit_map.get(tgt, "")
    st.metric(
        label=f"Solved: {tgt_label}",
        value=f"{tgt_val_display:.4f} {unit}".strip(),
    )

    # Contact force table
    force_col1, force_col2, force_col3, force_col4 = st.columns(4)
    force_col1.metric("N_A (Ground Normal)", f"{sol['N_A']:.2f} N")
    force_col2.metric("N_B (Wall Normal)",   f"{sol['N_B']:.2f} N")
    force_col3.metric("F_A (Ground Friction)", f"{sol['F_A']:.2f} N")
    force_col4.metric("F_B (Wall Friction)",   f"{sol['F_B']:.2f} N")

    # Step-by-step working
    with st.expander("📐 Step-by-step working", expanded=False):
        for step in sol.get("steps", []):
            st.text(step)

# ---------------------------------------------------------------------------
# Diagram (right column) – redrawn on every run from current inputs
# ---------------------------------------------------------------------------
def build_diagram(
    L: float,
    theta_deg: float,
    m_person: float,
    s: float,
    m_bucket: float,
    d: float,
    P: float,
    h_P: float,
    P_dir: int,
    solution: dict | None = None,
) -> plt.Figure:
    """Return a matplotlib Figure of the ladder geometry and (if solved) force arrows."""

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_aspect("equal")

    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    Bx = L * cos_t   # top of ladder (x)
    By = L * sin_t   # top of ladder (y)
    wall_x = Bx

    # --- Ground and wall ---
    ground_x = [-0.3, wall_x + 0.3]
    ax.plot(ground_x, [0, 0], "k-", lw=2, zorder=1)          # ground
    ax.plot([wall_x, wall_x], [0, By + 0.5], "k-", lw=3, zorder=1)  # wall
    ax.fill_between(ground_x, [-0.15, -0.15], [0, 0], color="lightgray", zorder=0)
    ax.fill_between([wall_x, wall_x + 0.15], [0, 0], [By + 0.5, By + 0.5],
                    color="lightgray", zorder=0)

    # --- Ladder ---
    ax.plot([0, Bx], [0, By], "saddlebrown", lw=4, solid_capstyle="round",
            label="Ladder", zorder=2)
    ax.plot(0, 0, "ko", ms=8, zorder=3)   # A (base)
    ax.plot(Bx, By, "ko", ms=8, zorder=3) # B (top)
    ax.annotate("A", (0, 0), textcoords="offset points", xytext=(-15, -12),
                fontsize=10, fontweight="bold")
    ax.annotate("B", (Bx, By), textcoords="offset points", xytext=(8, 4),
                fontsize=10, fontweight="bold")

    # --- Ladder midpoint weight (W_ladder) ---
    mid_x, mid_y = 0.5 * Bx, 0.5 * By
    ax.plot(mid_x, mid_y, "b^", ms=7, zorder=4)
    ax.annotate("W_ladder", (mid_x, mid_y), textcoords="offset points",
                xytext=(-55, 5), fontsize=8, color="blue")

    # Arrow scale: 1 N -> how many metres of arrow? scale to ~L/5 for a typical weight
    # We cap at L/3 so arrows don't overwhelm the diagram
    def arrow_scale(force_N: float, ref_N: float = 1000.0, max_len: float | None = None) -> float:
        max_l = max_len if max_len is not None else L / 3.0
        raw = (abs(force_N) / ref_N) * (L / 3.0)
        return min(raw, max_l) if abs(force_N) > 1e-6 else 0.0

    ARROW_KW = dict(width=0.025, head_width=0.12, head_length=0.08,
                    length_includes_head=True, zorder=5)

    # --- Person dot ---
    if m_person > 0:
        px = s * cos_t
        py = s * sin_t
        ax.plot(px, py, "go", ms=9, zorder=4, label=f"Person ({m_person:.0f} kg)")
        ax.annotate(f"W_p\n@s={s:.2f}m", (px, py),
                    textcoords="offset points", xytext=(8, -15),
                    fontsize=8, color="green")

    # --- Bucket dot ---
    if m_bucket > 0:
        bkx = d * cos_t
        bky = d * sin_t
        ax.plot(bkx, bky, "ms", ms=9, zorder=4, label=f"Bucket ({m_bucket:.0f} kg)")
        ax.annotate(f"W_b\n@d={d:.2f}m", (bkx, bky),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color="purple")

    # --- Force arrows (only after a successful solve) ---
    if solution is not None:
        N_A = solution["N_A"]
        N_B = solution["N_B"]
        F_A = solution["F_A"]
        F_B = solution["F_B"]

        ref = max(N_A, N_B, F_A, 1.0)  # normalise to largest contact force

        # N_A: upward at A
        len_NA = arrow_scale(N_A, ref)
        if len_NA > 0:
            ax.arrow(0, 0, 0, len_NA, color="royalblue", **ARROW_KW)
            ax.text(0.05, len_NA * 0.5, f"N_A\n{N_A:.0f}N", fontsize=7, color="royalblue")

        # F_A: toward wall (+x) at A
        len_FA = arrow_scale(F_A, ref)
        if len_FA > 0:
            ax.arrow(0, 0, len_FA, 0, color="darkcyan", **ARROW_KW)
            ax.text(len_FA * 0.4, -0.2, f"F_A\n{F_A:.0f}N", fontsize=7, color="darkcyan")

        # N_B: away from wall (-x) at B
        len_NB = arrow_scale(N_B, ref)
        if len_NB > 0:
            ax.arrow(Bx, By, -len_NB, 0, color="firebrick", **ARROW_KW)
            ax.text(Bx - len_NB - 0.1, By + 0.08,
                    f"N_B\n{N_B:.0f}N", fontsize=7, color="firebrick",
                    ha="right")

        # F_B: upward at B (wall friction, only if rough wall)
        len_FB = arrow_scale(F_B, ref)
        if len_FB > 0:
            ax.arrow(Bx, By, 0, len_FB, color="darkgreen", **ARROW_KW)
            ax.text(Bx + 0.05, By + len_FB * 0.5,
                    f"F_B\n{F_B:.0f}N", fontsize=7, color="darkgreen")

        # Applied force P at h_P along the ladder
        if P > 0:
            px_p = h_P * cos_t
            py_p = h_P * sin_t
            len_P = arrow_scale(P, ref)
            if len_P > 0:
                ax.arrow(px_p, py_p, P_dir * len_P, 0,
                         color="orange", **ARROW_KW)
                ax.text(px_p + P_dir * len_P * 0.5, py_p + 0.12,
                        f"P\n{P:.0f}N", fontsize=7, color="orange", ha="center")

    # --- Angle annotation ---
    angle_arc = mpatches.Arc((0, 0), 0.6, 0.6, angle=0,
                              theta1=0, theta2=theta_deg, color="gray", lw=1.2)
    ax.add_patch(angle_arc)
    ax.text(0.38, 0.08, f"θ={theta_deg:.1f}°", fontsize=8, color="gray")

    # --- Axis formatting ---
    margin = 0.5
    ax.set_xlim(-0.5, wall_x + 0.5)
    ax.set_ylim(-0.4, By + margin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    title_str = "Ladder Geometry"
    if solution is not None:
        title_str += " + Contact Forces"
    ax.set_title(title_str, fontsize=11)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig


# Render diagram in the right column, using current UI state
with right_col:
    st.subheader("Diagram")

    # Determine theta for diagram: if target is theta and solved, use solved value; else use input
    diag_theta = theta_val
    diag_s = s_val
    if last is not None:
        sol_snap = last["sol"]
        if last["target"] == "theta_deg":
            diag_theta = sol_snap["theta_deg"]
        if last["target"] == "s":
            diag_s = sol_snap["s"]

    diag_solution = last["sol"] if last is not None else None

    fig = build_diagram(
        L=L_val if target != "L" else (last["sol"]["L"] if last else 5.0),
        theta_deg=diag_theta,
        m_person=m_person_val if target != "m_person" else (last["sol"]["m_person"] if last else 0.0),
        s=diag_s,
        m_bucket=m_bucket_val,
        d=d_val,
        P=P_val if target != "P" else (last["sol"]["P"] if last else 0.0),
        h_P=h_P_val if target != "h_P" else (last["sol"]["h_P"] if last else 0.0),
        P_dir=P_dir_val,
        solution=diag_solution,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
