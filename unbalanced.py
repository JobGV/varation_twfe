"""Checks Proposition 3 for unbalanced panels and an explicit counterexample."""
from __future__ import annotations

import numpy as np
from numpy.linalg import pinv

RNG = np.random.default_rng(11)


def treatment_matrix(G: np.ndarray, T: int) -> np.ndarray:
    return np.array([[float(t + 1 >= g) for t in range(T)] for g in G])


def connected(O: np.ndarray) -> bool:
    N, T = O.shape
    if O.sum(axis=1).min() < 1 or O.sum(axis=0).min() < 1:
        return False
    seen_i, seen_t = set(), set()
    stack = [("i", 0)]
    while stack:
        kind, idx = stack.pop()
        if kind == "i":
            if idx in seen_i:
                continue
            seen_i.add(idx)
            stack.extend(("t", t) for t in range(T) if O[idx, t])
        else:
            if idx in seen_t:
                continue
            seen_t.add(idx)
            stack.extend(("i", i) for i in range(N) if O[i, idx])
    return len(seen_i) == N and len(seen_t) == T


def V_projection(D: np.ndarray, O: np.ndarray) -> float:
    """Brute-force two-way projection on observed cells using an SVD."""
    N, T = O.shape
    cells = [(i, t) for i in range(N) for t in range(T) if O[i, t]]
    M = len(cells)
    X = np.zeros((M, N + T))
    d = np.zeros(M)
    for row, (i, t) in enumerate(cells):
        X[row, i] = 1.0
        X[row, N + t] = 1.0
        d[row] = D[i, t]
    U, singular_values, _ = np.linalg.svd(X, full_matrices=False)
    keep = singular_values > singular_values.max() * 1e-10
    residual = d - U[:, keep] @ (U[:, keep].T @ d)
    return float(residual @ residual / M)


def V_semiclosed(D: np.ndarray, O: np.ndarray) -> float:
    """Proposition 3; requires only a T-by-T pseudoinverse."""
    N, T = O.shape
    Ti = O.sum(axis=1).astype(float)
    si = (D * O).sum(axis=1).astype(float)
    M = int(O.sum())
    first = np.sum(si * (Ti - si) / Ti)

    q = np.array([
        sum(D[i, t] - si[i] / Ti[i] for i in range(N) if O[i, t])
        for t in range(T)
    ])
    Nt = O.sum(axis=0).astype(float)
    A = np.zeros((T, T))
    for i in range(N):
        obs = np.flatnonzero(O[i])
        A[np.ix_(obs, obs)] += 1.0 / Ti[i]
    gram = np.diag(Nt) - A
    return float((first - q @ pinv(gram, rcond=1e-9) @ q) / M)


print("=" * 72)
print("PROPOSITION 3: semi-closed form versus brute-force projection")
print("=" * 72)
max_diff = 0.0
for _ in range(150):
    T = int(RNG.integers(4, 10))
    N = int(RNG.integers(20, 70))
    G = RNG.choice(list(range(2, T + 1)) + [np.inf], size=N)
    D = treatment_matrix(G, T)
    O = (RNG.random((N, T)) < RNG.uniform(0.55, 0.95)).astype(int)
    if not connected(O):
        continue
    max_diff = max(max_diff, abs(V_projection(D, O) - V_semiclosed(D, O)))
print(f"maximum absolute difference: {max_diff:.3e}")
assert max_diff < 1e-10
print("PASS\n")

print("=" * 72)
print("EXPLICIT FAILURE OF THE BALANCED PER-CELL BOUND")
print("=" * 72)
G = np.array([2.0, 3.0, 4.0])
D = treatment_matrix(G, T=4)
O = np.array([
    [1, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 1, 1],
], dtype=int)
O_balanced = np.ones_like(O)

v_unbalanced = V_semiclosed(D, O)
v_balanced = V_semiclosed(D, O_balanced)
M = int(O.sum())
NT = int(D.size)
print(f"V unbalanced = {v_unbalanced:.6f} > 1/12 = {1/12:.6f}")
print(f"V balanced   = {v_balanced:.6f}")
print(f"M V unbalanced  = {M*v_unbalanced:.6f}")
print(f"NT V balanced   = {NT*v_balanced:.6f}")
print(f"information ratio = {(M*v_unbalanced)/(NT*v_balanced):.3f}")
assert abs(v_unbalanced - 0.1) < 1e-12
assert abs(v_balanced - 5 / 72) < 1e-12
assert M * v_unbalanced < NT * v_balanced
print("PASS")
