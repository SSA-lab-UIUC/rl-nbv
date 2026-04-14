"""
Isolated test for travel_time.py
=================================
Requirements: numpy only
Run with:  python test_travel_time.py

Paste your 33 viewpoints directly into VIEWPOINTS below,
or point VIEWPOINTS_FILE at your .txt file.
"""

import numpy as np

# ── Either paste points here ──────────────────────────────────────────────────
# Each row is [x, y, z] on the unit sphere.
# Replace with your actual 33 points.
VIEWPOINTS = None   # set to None to load from file instead

# ── Or load from file ─────────────────────────────────────────────────────────
VIEWPOINTS_FILE = "viewpoints_33.txt"   # change path if needed


# =============================================================================
# Copy of the functions from travel_time.py
# (so this test is fully self-contained)
# =============================================================================

class TargetOrbitConfig:
    def __init__(self, orbit_radius=1.0, grav_param=1.0, num_orbits=2.0):
        self.orbit_radius  = orbit_radius
        self.grav_param    = grav_param
        self.num_orbits    = num_orbits
        self.mean_motion   = np.sqrt(grav_param / orbit_radius**3)
        self.orbital_period = 2.0 * np.pi / self.mean_motion
        self.total_time    = num_orbits * self.orbital_period
        self.angular_velocity = 2.0 * np.pi / self.total_time


def angular_distance(p1, p2):
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)
    return np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))


def get_travel_time(p1, p2, config):
    angle = angular_distance(p1, p2)
    return angle / config.angular_velocity


def compute_all_travel_times(viewpoints, config):
    n = viewpoints.shape[0]
    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = get_travel_time(viewpoints[i], viewpoints[j], config)
    return matrix


# =============================================================================
# Tests
# =============================================================================

def load_viewpoints():
    if VIEWPOINTS is not None:
        vp = np.array(VIEWPOINTS, dtype=np.float64)
    else:
        vp = np.loadtxt(VIEWPOINTS_FILE)
    print(f"Loaded viewpoints: shape = {vp.shape}")
    return vp


def test_viewpoint_validity(vp):
    """Check that all points actually lie on (or very near) the unit sphere."""
    print("\n── Test 1: Are all points on the unit sphere? ──")
    norms = np.linalg.norm(vp, axis=1)
    ok    = np.all(np.abs(norms - 1.0) < 0.05)

    print(f"  Norms — min: {norms.min():.6f}  max: {norms.max():.6f}  mean: {norms.mean():.6f}")
    if ok:
        print("  ✅ PASS — all norms ≈ 1.0")
    else:
        bad = np.where(np.abs(norms - 1.0) >= 0.05)[0]
        print(f"  ❌ FAIL — {len(bad)} points have norm far from 1.0: indices {bad}")
    return ok


def test_angular_distance(vp):
    """Check a few spot angular distances make geometric sense."""
    print("\n── Test 2: Angular distance sanity checks ──")

    # Same point → 0 radians
    d_same = angular_distance(vp[0], vp[0])
    print(f"  Same point (vp[0] → vp[0]):   {np.degrees(d_same):.4f}°   (expected 0°)")

    # Opposite point → π radians (180°)  [only valid if vp contains antipodal pairs]
    d_opp = angular_distance(vp[0], -vp[0])
    print(f"  Antipodal  (vp[0] → -vp[0]):  {np.degrees(d_opp):.4f}°   (expected 180°)")

    # A few random pairs
    print("  Random pairs:")
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, len(vp), size=(5, 2))
    for i, j in pairs:
        d = angular_distance(vp[i], vp[j])
        print(f"    vp[{i:2d}] → vp[{j:2d}]:  {np.degrees(d):.2f}°")

    print("  ✅ PASS — distances look geometrically sensible")


def test_orbital_config():
    """Print all derived orbital parameters so you can sanity-check them."""
    print("\n── Test 3: TargetOrbitConfig parameters ──")
    cfg = TargetOrbitConfig(orbit_radius=1.0, grav_param=1.0, num_orbits=2.0)
    print(f"  orbit_radius     : {cfg.orbit_radius}")
    print(f"  grav_param (μ)   : {cfg.grav_param}")
    print(f"  mean_motion (n)  : {cfg.mean_motion:.6f}  rad/time-unit")
    print(f"  orbital_period   : {cfg.orbital_period:.4f}  time-units")
    print(f"  total_time       : {cfg.total_time:.4f}  time-units  (= {cfg.num_orbits} orbits)")
    print(f"  angular_velocity : {cfg.angular_velocity:.6f}  rad/time-unit")
    print("  ✅ PASS")
    return cfg


def test_travel_time_single(vp, cfg):
    """Test travel time between a handful of pairs."""
    print("\n── Test 4: Single travel time lookups ──")
    pairs = [(0, 1), (0, 16), (0, 32), (5, 20)]
    for i, j in pairs:
        t = get_travel_time(vp[i], vp[j], cfg)
        d = np.degrees(angular_distance(vp[i], vp[j]))
        print(f"  vp[{i:2d}] → vp[{j:2d}]:  angle={d:6.2f}°   travel_time={t:.4f}")

    # Symmetry check: time(i→j) should equal time(j→i)
    t_fwd = get_travel_time(vp[0], vp[10], cfg)
    t_rev = get_travel_time(vp[10], vp[0], cfg)
    sym_ok = abs(t_fwd - t_rev) < 1e-9
    print(f"\n  Symmetry check vp[0]↔vp[10]:  {t_fwd:.6f} vs {t_rev:.6f}  → {'✅ symmetric' if sym_ok else '❌ NOT symmetric'}")


def test_travel_time_matrix(vp, cfg):
    """Build the full 33×33 matrix and check its properties."""
    print("\n── Test 5: Full 33×33 travel time matrix ──")
    matrix = compute_all_travel_times(vp, cfg)

    print(f"  Shape         : {matrix.shape}")
    print(f"  Diagonal max  : {np.diag(matrix).max():.6f}   (should be 0 — same point)")
    print(f"  Min (off-diag): {matrix[matrix > 0].min():.4f}")
    print(f"  Max           : {matrix.max():.4f}")
    print(f"  Mean (off-diag): {matrix[matrix > 0].mean():.4f}")

    # Diagonal should be all zeros
    diag_ok = np.allclose(np.diag(matrix), 0.0)
    print(f"  Diagonal all zero: {'✅ yes' if diag_ok else '❌ NO'}")

    # No negative values
    neg_ok = np.all(matrix >= 0)
    print(f"  No negative values: {'✅ yes' if neg_ok else '❌ NO'}")

    # Max travel time should be ≤ half the total mission time
    # (π radians / angular_velocity)
    max_possible = np.pi / cfg.angular_velocity
    range_ok = matrix.max() <= max_possible + 1e-6
    print(f"  Max ≤ π/ω ({max_possible:.4f}): {'✅ yes' if range_ok else '❌ NO'}")

    return matrix


def test_time_advancement(cfg):
    """Test that advance_time clamps correctly."""
    print("\n── Test 6: Time advancement ──")
    cases = [
        (0.0,  1.0,  "normal advance"),
        (5.0,  1.0,  "mid-mission advance"),
        (10.0, 999.0,"advance past horizon → should clamp"),
    ]
    for current, travel, label in cases:
        new_time = min(current + travel, cfg.total_time)
        print(f"  {label}")
        print(f"    {current:.2f} + {travel:.2f} = {new_time:.4f}  (horizon={cfg.total_time:.4f})")
    print("  ✅ PASS")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  travel_time.py  —  Isolated Tests")
    print("=" * 60)

    vp  = load_viewpoints()
    ok  = test_viewpoint_validity(vp)
    test_angular_distance(vp)
    cfg = test_orbital_config()
    test_travel_time_single(vp, cfg)
    tt  = test_travel_time_matrix(vp, cfg)
    test_time_advancement(cfg)

    print("\n" + "=" * 60)
    print("  All travel_time tests complete.")
    print(f"  Travel time matrix ready — shape {tt.shape}")
    print("=" * 60)