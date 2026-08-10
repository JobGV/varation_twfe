"""Replication checks for "How much variation does two-way fixed effects use?"

Verifies the balanced-panel closed form (Lemma 1), the exact finite-T
decomposition and bound (Proposition 1), the discrete optimal rollout under a
fixed eventual treated share (Proposition 2), and the calibration table.
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20260804)


def treatment_matrix(G: np.ndarray, T: int) -> np.ndarray:
    """Absorbing treatment matrix; np.inf denotes never treated."""
    return np.array([[float(t + 1 >= g) for t in range(T)] for g in G])


def V_direct(G: np.ndarray, T: int) -> float:
    """V from explicit two-way residualization on a balanced panel."""
    D = treatment_matrix(G, T)
    Dt = D - D.mean(axis=1, keepdims=True) - D.mean(axis=0, keepdims=True) + D.mean()
    return float(np.mean(Dt**2))


def V_closed(G: np.ndarray, T: int) -> float:
    """Lemma 1: calendar representation of V."""
    F = np.array([(G <= t + 1).mean() for t in range(T)])
    s = np.array([T - g + 1 if np.isfinite(g) else 0.0 for g in G])
    return float(np.mean(F * (1.0 - F)) - np.var(s) / T**2)


def V_shares(pi: np.ndarray) -> float:
    """V from cohort shares.

    pi has length T: pi[0] is never treated and pi[g-1] adopts in period g,
    g=2,...,T.
    """
    T = len(pi)
    F = treated_shares(pi)
    durations = np.array([0.0] + list(range(T - 1, 0, -1)))
    var_s = pi @ durations**2 - (pi @ durations) ** 2
    return float(np.mean(F * (1.0 - F)) - var_s / T**2)


def bound(T: int) -> float:
    return (T**2 - 1.0) / (12.0 * T**2)


def zgrid(T: int) -> np.ndarray:
    return (2 * np.arange(1, T + 1) - T - 1) / T


def square_completion(F: np.ndarray) -> float:
    """Proposition 1 right-hand side."""
    T = len(F)
    return float(bound(T) - np.var(F - zgrid(T) / 2.0))


def treated_shares(pi: np.ndarray) -> np.ndarray:
    T = len(pi)
    F = np.zeros(T)
    for t in range(2, T + 1):
        F[t - 1] = pi[1:t].sum()
    return F


def V_from_F(F: np.ndarray) -> float:
    return square_completion(F)


# ---------------------------------------------------------------------------
# Proposition 2: discrete optimal rollout for a fixed eventual treated share.
# Write h_t = F_t - z_t/2.  F_1 = 0 and F_T = a pin h_1 and h_T; monotonicity
# of F is h_{t+1} - h_t >= -1/T.  The optimum clips a constant to the
# reachability corridor [L_t, U_t] and leaves the two endpoints at their
# forced values.
# ---------------------------------------------------------------------------
def corridor(T: int, a: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    z = zgrid(T)
    h1, hT = -z[0] / 2.0, a - z[-1] / 2.0
    t = np.arange(1, T + 1)
    return h1 - (t - 1) / T, hT + (T - t) / T, h1, hT


def discrete_optimum(T: int, a: float) -> tuple[np.ndarray, float, float]:
    """Return (cohort shares pi, V, level m) for the fixed-a optimum."""
    L, U, h1, hT = corridor(T, a)
    lo, hi = float(min(L.min(), hT)), float(max(U.max(), h1))
    for _ in range(200):                      # bisection on m = mean(h*(m))
        m = 0.5 * (lo + hi)
        h = np.clip(m, L, U)
        h[0], h[-1] = h1, hT
        if h.mean() > m:
            lo = m
        else:
            hi = m
    m = 0.5 * (lo + hi)
    h = np.clip(m, L, U)
    h[0], h[-1] = h1, hT
    F = h + zgrid(T) / 2.0
    pi = np.diff(np.concatenate(([0.0], F)))
    pi[0] = 1.0 - a                            # never-treated share
    return pi, float(bound(T) - np.var(h)), float(m)


def V_continuous(a: float) -> float:
    """Continuous-period value V*(a) = a(a^2 - 3a + 3)/12."""
    return a * (a**2 - 3 * a + 3) / 12.0


print("=" * 72)
print("LEMMA 1: closed form versus direct two-way projection")
print("=" * 72)
max_diff = 0.0
for _ in range(300):
    T = int(RNG.integers(3, 16))
    N = int(RNG.integers(25, 180))
    G = RNG.choice(list(range(2, T + 1)) + [np.inf], size=N)
    max_diff = max(max_diff, abs(V_direct(G, T) - V_closed(G, T)))
print(f"maximum absolute difference: {max_diff:.3e}")
assert max_diff < 1e-12
print("PASS\n")

print("=" * 72)
print("PROPOSITION 1: exact finite-T square completion and bound")
print("=" * 72)
max_diff = 0.0
max_violation = -np.inf
for _ in range(500):
    T = int(RNG.integers(3, 25))
    pi = RNG.dirichlet(np.ones(T))
    F = treated_shares(pi)
    v = V_shares(pi)
    max_diff = max(max_diff, abs(v - square_completion(F)))
    max_violation = max(max_violation, v - bound(T))
print(f"maximum identity error: {max_diff:.3e}")
print(f"largest V - finite bound: {max_violation:.3e}")
assert max_diff < 1e-12 and max_violation <= 1e-12

for T in [4, 6, 10, 20]:
    pi_star = np.full(T, 1.0 / T)
    v = V_shares(pi_star)
    print(f"T={T:2d}: V(pi*)={v:.9f}, bound={bound(T):.9f}")
    assert abs(v - bound(T)) < 1e-12
print("PASS\n")

print("=" * 72)
print("PROPOSITION 2: discrete optimum under a fixed eventual treated share")
print("=" * 72)

# (i) KKT verification against a brute-force optimizer, when SciPy is present.
try:
    from scipy.optimize import minimize

    worst = 0.0
    for T, a in [(20, 0.6), (10, 0.6), (8, 0.5), (12, 0.9), (10, 1.0),
                 (20, 1.0), (6, 1.0), (10, 0.8), (15, 0.35), (9, 0.25)]:
        _, v_star, _ = discrete_optimum(T, a)
        z = zgrid(T)
        obj = lambda p: np.var(np.cumsum(np.concatenate(([0.0], p))) - z / 2)
        best = None
        for seed in range(15):
            x0 = np.random.default_rng(seed).dirichlet(np.ones(T - 1)) * a
            r = minimize(obj, x0, bounds=[(0, 1)] * (T - 1),
                         constraints=[{"type": "eq", "fun": lambda p: p.sum() - a}],
                         method="SLSQP", options={"maxiter": 3000, "ftol": 1e-16})
            best = r if best is None or r.fun < best.fun else best
        worst = max(worst, (bound(T) - best.fun) - v_star)
    print(f"largest numerical improvement over the closed form: {worst:.3e}")
    assert worst < 1e-9
except ImportError:                            # pragma: no cover
    print("SciPy unavailable; skipping the brute-force cross-check")

# (ii) The a = 1 corner: U-shaped optimum with a closed form.
print(f"\n{'T':>4}{'equal cohorts':>16}{'% bound':>10}{'U-shaped opt':>15}{'% bound':>10}")
for T in [6, 10, 20]:
    F_eq = np.concatenate(([0.0], np.arange(1, T) / (T - 1)))
    pi, v_opt, _ = discrete_optimum(T, 1.0)
    v_eq = V_from_F(F_eq)
    assert abs(v_opt - (bound(T) - 1.0 / (2 * T**3))) < 1e-12
    assert abs(pi[1] - 1.5 / T) < 1e-12 and abs(pi[-1] - 1.5 / T) < 1e-12
    assert np.allclose(pi[2:-1], 1.0 / T)
    print(f"{T:4d}{v_eq:16.6f}{100*v_eq/bound(T):9.2f}%{v_opt:15.6f}"
          f"{100*v_opt/bound(T):9.2f}%")

# (iii) The calibration reported in the text.
T, a = 20, 0.60
pi_opt, v_opt, m = discrete_optimum(T, a)
pi_uniform = np.zeros(T)
pi_uniform[0] = 1 - a
pi_uniform[1:] = a / (T - 1)
v_uniform = V_shares(pi_uniform)
print(f"\nT={T}, a={a}: level m={m:.6f}")
print(f"  nonzero cohort shares: pi_inf={pi_opt[0]:.4f}, "
      f"pi_5=pi_17={pi_opt[4]:.4f}, pi_6..pi_16={pi_opt[5]:.4f}")
print(f"  discrete optimum       V={v_opt:.6f}")
print(f"  equal over g=2,...,T   V={v_uniform:.6f}")
print(f"  continuous limit V*(a)={V_continuous(a):.6f}")
print(f"  relative improvement: {100*(v_opt/v_uniform-1):.2f}%")
assert abs(v_opt - 0.077875) < 1e-12
assert abs(v_uniform - 0.0718421052631579) < 1e-12
assert abs(V_shares(pi_opt) - v_opt) < 1e-12
print("PASS\n")

print("=" * 72)
print("TABLE 1: selected designs")
print("=" * 72)


def design(T: int, never: float, cohorts: dict[int, float]) -> np.ndarray:
    pi = np.zeros(T)
    pi[0] = never
    for g, share in cohorts.items():
        pi[g - 1] = share
    if not np.isclose(pi.sum(), 1.0):
        raise ValueError(f"cohort shares sum to {pi.sum()}, not one")
    return pi


designs = [
    ("Optimal cohort shares", 10, np.full(10, 0.1)),
    ("Optimal with no never-treated group", 10, discrete_optimum(10, 1.0)[0]),
    ("9 equal adoption cohorts, no never", 10,
     design(10, 0.0, {g: 1 / 9 for g in range(2, 11)})),
    ("9 equal cohorts, 40% never", 10,
     design(10, 0.4, {g: 1 / 15 for g in range(2, 11)})),
    ("50% never; 30% g=3, 20% g=5", 6,
     design(6, 0.5, {3: 0.3, 5: 0.2})),
    ("20% never; 80% g=5", 10,
     design(10, 0.2, {5: 0.8})),
    ("40% never; 30% each g=9,10", 10,
     design(10, 0.4, {9: 0.3, 10: 0.3})),
]
print(f"{'design':<42}{'T':>4}{'V':>11}{'% bound':>11}")
for name, T, pi in designs:
    v = V_shares(pi)
    print(f"{name:<42}{T:4d}{v:11.6f}{100*v/bound(T):10.1f}%")
print("\nPASS")
