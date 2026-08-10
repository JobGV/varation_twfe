"""Numerical check of the correspondence with Hussey and Hughes (2007), Eq. (8)."""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(7)


def V_calendar(G: np.ndarray, T: int) -> float:
    F = np.array([(G <= t + 1).mean() for t in range(T)])
    s = np.array([T - g + 1 if np.isfinite(g) else 0.0 for g in G])
    return float(np.mean(F * (1 - F)) - np.var(s) / T**2)


def hh_tau_coefficient(G: np.ndarray, T: int) -> float:
    """U^2 + I T U - T W - I V_HH in Hussey--Hughes notation."""
    I = len(G)
    X = np.array([[float(t + 1 >= G[i]) for t in range(T)] for i in range(I)])
    U = X.sum()
    W = np.sum(X.sum(axis=0) ** 2)
    V_hh = np.sum(X.sum(axis=1) ** 2)
    return float(U**2 + I * T * U - T * W - I * V_hh)


print(f"{'N':>5}{'T':>4}{'(NT)^2 V':>18}{'HH coefficient':>18}{'relative error':>17}")
worst = 0.0
for _ in range(50):
    T = int(RNG.integers(3, 12))
    N = int(RNG.integers(10, 80))
    G = RNG.choice(list(range(2, T + 1)) + [np.inf], size=N)
    lhs = (N * T) ** 2 * V_calendar(G, T)
    rhs = hh_tau_coefficient(G, T)
    relative_error = abs(lhs - rhs) / max(abs(lhs), 1e-12)
    worst = max(worst, relative_error)
    if _ < 12:
        print(f"{N:5d}{T:4d}{lhs:18.6f}{rhs:18.6f}{relative_error:17.2e}")
print(f"\nworst relative error over 50 designs: {worst:.2e}")
assert worst < 1e-12
print("PASS")
