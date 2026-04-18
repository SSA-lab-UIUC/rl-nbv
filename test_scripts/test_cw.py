"""
Isolated test for cw_utils.py
===============================
Requirements: numpy only
Run with:  python test_cw_utils.py

This test script:
  1. Verifies the CW state-transition matrix is mathematically correct.
  2. Checks that propagating with the solved v0 actually lands at rf.
  3. Builds the full 33×33 Δv matrix and reports statistics.
  4. Flags any singular (infeasible) transfers.
"""

import numpy as np

# ── Point to your viewpoints ──────────────────────────────────────────────────
VIEWPOINTS_FILE = "viewpoints_33.txt"   # change if needed
ORBIT_RADIUS    = 1.0
GRAV_PARAM      = 1.0
NUM_ORBITS      = 2.0


# =============================================================================
# Inline copy of the classes so test is fully self-contained
# =============================================================================

def _build_stm(n, t):
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)

    Phi_rr = np.array([
        [4 - 3*c,       0,  0],
        [6*(s - nt),    1,  0],
        [0,             0,  c],
    ])
    Phi_rv = np.array([
        [ s/n,          2*(1 - c)/n,     0  ],
        [-2*(1 - c)/n, (4*s - 3*nt)/n,   0  ],
        [ 0,            0,               s/n],
    ])
    Phi_vr = np.array([
        [3*n*s,       0,  0    ],
        [6*n*(c - 1), 0,  0    ],
        [0,           0, -n*s  ],
    ])
    Phi_vv = np.array([
        [ c,    2*s,      0],
        [-2*s,  4*c - 3,  0],
        [ 0,    0,        c],
    ])
    return Phi_rr, Phi_rv, Phi_vr, Phi_vv


class CWDynamics:
    def __init__(self, mean_motion):
        self.n = mean_motion

    def compute_delta_v(self, r0, rf, t):
        if t <= 0.0:
            return 0.0, np.zeros(3)
        Phi_rr, Phi_rv, _, _ = _build_stm(self.n, t)
        rhs = rf - Phi_rr @ r0
        try:
            v0 = np.linalg.solve(Phi_rv, rhs)
        except np.linalg.LinAlgError:
            return np.inf, None
        return float(np.linalg.norm(v0)), v0

    def propagate(self, r0, v0, t):
        Phi_rr, Phi_rv, Phi_vr, Phi_vv = _build_stm(self.n, t)
        rf = Phi_rr @ r0 + Phi_rv @ v0
        vf = Phi_vr @ r0 + Phi_vv @ v0
        return rf, vf


class TargetOrbitConfig:
    def __init__(self, orbit_radius=1.0, grav_param=1.0, num_orbits=2.0):
        self.orbit_radius     = orbit_radius
        self.mean_motion      = np.sqrt(grav_param / orbit_radius**3)
        self.orbital_period   = 2.0 * np.pi / self.mean_motion
        self.total_time       = num_orbits * self.orbital_period
        self.angular_velocity = 2.0 * np.pi / self.total_time


def angular_distance(p1, p2):
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    return np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))


def get_travel_time(p1, p2, cfg):
    return angular_distance(p1, p2) / cfg.angular_velocity


# =============================================================================
# Tests
# =============================================================================

def load_viewpoints():
    vp = np.loadtxt(VIEWPOINTS_FILE)
    print(f"Loaded viewpoints: {vp.shape}")
    return vp


def test_stm_identity(cfg):
    """
    At t=0 the STM should be the identity:
      Phi_rr = I,  Phi_rv = 0,  Phi_vr = 0,  Phi_vv = I
    (A very small t ≈ 0 is used because t=0 exactly gives 0/0 in s/n.)
    """
    print("\n── Test 1: STM approaches identity as t → 0 ──")
    t_tiny = 1e-9
    Phi_rr, Phi_rv, Phi_vr, Phi_vv = _build_stm(cfg.mean_motion, t_tiny)

    rr_ok = np.allclose(Phi_rr, np.eye(3), atol=1e-4)
    rv_ok = np.allclose(Phi_rv, np.zeros((3, 3)), atol=1e-4)
    vv_ok = np.allclose(Phi_vv, np.eye(3), atol=1e-4)

    print(f"  Phi_rr ≈ I   : {'✅' if rr_ok else '❌'}")
    print(f"  Phi_rv ≈ 0   : {'✅' if rv_ok else '❌'}")
    print(f"  Phi_vv ≈ I   : {'✅' if vv_ok else '❌'}")


def test_round_trip(cfg):
    """
    Core correctness test: solve for v0, propagate forward, check we land at rf.
    If this passes, the CW solver is working correctly.
    """
    print("\n── Test 2: Round-trip accuracy (solve v0, propagate, check rf) ──")
    cw = CWDynamics(cfg.mean_motion)

    test_cases = [
        # (r0,                          rf,                          label)
        (np.array([1., 0., 0.]),  np.array([0., 1., 0.]),  "90° move (x→y)"),
        (np.array([1., 0., 0.]),  np.array([-1., 0., 0.]), "180° move (x→-x)"),
        (np.array([0., 1., 0.]),  np.array([0., 0., 1.]),  "90° move (y→z)"),
        (np.array([0.6, 0.8, 0.]),np.array([0., 0., 1.]),  "arbitrary move"),
    ]

    t = cfg.orbital_period / 4.0    # quarter-orbit time of flight
    print(f"  Time of flight used: {t:.4f} time-units (= quarter orbit)")

    all_ok = True
    for r0, rf, label in test_cases:
        dv, v0 = cw.compute_delta_v(r0 * ORBIT_RADIUS, rf * ORBIT_RADIUS, t)
        if v0 is None:
            print(f"  {label}: ❌ SINGULAR — transfer infeasible at this time")
            all_ok = False
            continue
        rf_check, _ = cw.propagate(r0 * ORBIT_RADIUS, v0, t)
        error = np.linalg.norm(rf_check - rf * ORBIT_RADIUS)
        ok    = error < 1e-6
        all_ok = all_ok and ok
        print(f"  {label}:  Δv={dv:.4f}   position_error={error:.2e}   {'✅' if ok else '❌'}")

    if all_ok:
        print("  ✅ All round-trips passed — CW solver is correct")
    else:
        print("  ❌ Some round-trips failed — check your orbital parameters")


def test_zero_distance(cfg):
    """Travel from a point to itself — Δv should be 0."""
    print("\n── Test 3: Zero-distance move (same viewpoint) ──")
    cw = CWDynamics(cfg.mean_motion)
    r  = np.array([1.0, 0.0, 0.0]) * ORBIT_RADIUS
    t  = cfg.orbital_period / 4.0
    dv, v0 = cw.compute_delta_v(r, r, t)
    ok = dv < 1e-6
    print(f"  Δv = {dv:.2e}   {'✅ ≈ 0 as expected' if ok else '❌ should be ~0'}")


def test_delta_v_matrix(vp, cfg):
    """
    Build the full 33×33 Δv matrix using actual viewpoints and travel times.
    Reports statistics and flags any infeasible transfers.
    """
    print("\n── Test 4: Full 33×33 Δv matrix ──")
    cw = CWDynamics(cfg.mean_motion)
    n  = vp.shape[0]

    # First build the travel time matrix
    tt = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            tt[i, j] = get_travel_time(vp[i], vp[j], cfg)

    # Now build the Δv matrix
    dv_matrix = np.zeros((n, n), dtype=np.float32)
    inf_count  = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                dv_matrix[i, j] = 0.0
                continue
            r0 = vp[i] * ORBIT_RADIUS
            rf = vp[j] * ORBIT_RADIUS
            t  = float(tt[i, j])
            dv, _ = cw.compute_delta_v(r0, rf, t)
            dv_matrix[i, j] = dv
            if not np.isfinite(dv):
                inf_count += 1

    finite_vals = dv_matrix[np.isfinite(dv_matrix) & (dv_matrix > 0)]

    print(f"  Matrix shape      : {dv_matrix.shape}")
    print(f"  Diagonal max      : {np.diag(dv_matrix).max():.6f}  (should be 0)")
    print(f"  Min Δv (off-diag) : {finite_vals.min():.4f}")
    print(f"  Max Δv            : {finite_vals.max():.4f}")
    print(f"  Mean Δv           : {finite_vals.mean():.4f}")
    print(f"  Infeasible pairs  : {inf_count}  {'✅ none' if inf_count == 0 else '⚠️  check these'}")

    # Show the 5 cheapest and 5 most expensive transfers
    print("\n  5 cheapest transfers:")
    flat = [(dv_matrix[i, j], i, j) for i in range(n) for j in range(n) if i != j and np.isfinite(dv_matrix[i, j])]
    flat.sort()
    for dv, i, j in flat[:5]:
        print(f"    vp[{i:2d}] → vp[{j:2d}]:  Δv = {dv:.4f}")

    print("\n  5 most expensive transfers:")
    for dv, i, j in flat[-5:][::-1]:
        print(f"    vp[{i:2d}] → vp[{j:2d}]:  Δv = {dv:.4f}")

    return dv_matrix, tt


def test_fuel_budget_feasibility(dv_matrix):
    """
    Given a fuel budget of 50.0, how many single-hop transfers are affordable?
    Gives you a sense of whether the budget is too tight or too generous.
    """
    print("\n── Test 5: Fuel budget feasibility check ──")
    budget = 50.0
    n = dv_matrix.shape[0]
    finite = dv_matrix[np.isfinite(dv_matrix) & (dv_matrix > 0)]

    affordable = np.sum(finite <= budget)
    total      = len(finite)

    print(f"  Fuel budget       : {budget}")
    print(f"  Affordable hops   : {affordable} / {total}  ({100*affordable/total:.1f}%)")
    print(f"  Max hops possible : ~{int(budget // finite.mean())}  (budget / mean Δv)")

    if affordable == total:
        print("  ✅ All single hops affordable — budget may be too generous")
    elif affordable > total * 0.8:
        print("  ✅ Most hops affordable — budget looks reasonable")
    else:
        print("  ⚠️  Many hops unaffordable — consider increasing fuel_budget")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  cw_utils.py  —  Isolated Tests")
    print("=" * 60)

    vp  = load_viewpoints()
    cfg = TargetOrbitConfig(ORBIT_RADIUS, GRAV_PARAM, NUM_ORBITS)

    print(f"\n  mean_motion      : {cfg.mean_motion:.6f}")
    print(f"  orbital_period   : {cfg.orbital_period:.4f}")
    print(f"  total_time       : {cfg.total_time:.4f}")

    test_stm_identity(cfg)
    test_round_trip(cfg)
    test_zero_distance(cfg)
    dv_matrix, tt = test_delta_v_matrix(vp, cfg)
    test_fuel_budget_feasibility(dv_matrix)

    print("\n" + "=" * 60)
    print("  All cw_utils tests complete.")
    print("=" * 60)