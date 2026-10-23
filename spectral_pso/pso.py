"""Particle Swarm Optimisation over LED mixing weights on the probability simplex."""

import numpy as np


def project_to_simplex(x):
    """Clip negatives to zero and rescale rows to sum to one.

    This is the projection the lab used (not the Euclidean one): an LED cannot
    emit negative light, and only the relative mix matters for colour because
    XYZ is normalised to the light's own white. Rows that clip to all zeros fall
    back to an equal mix so the particle stays valid.
    """
    x = np.maximum(x, 0.0)
    total = x.sum(axis=-1, keepdims=True)
    uniform = np.full_like(x, 1.0 / x.shape[-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, x / total, uniform)


def optimise(cost_fn, n_dims, maximise=False, n_particles=200, n_iterations=200,
             inertia=0.95, c_personal=5.0, c_global=5.0, seed=None, verbose=True):
    """Find simplex weights that minimise (or maximise) cost_fn.

    cost_fn takes weights shaped (P, n_dims) and returns costs shaped (P,);
    NaN marks an infeasible candidate (for example an out-of-gamut colour).
    Defaults match the 2025 run: 200 particles, 200 iterations, c0 = 0.95,
    c1 = c2 = 5. Updates are synchronous: every particle sees the global best
    from the end of the previous iteration.

    Returns (best_weights, best_cost, history) where history holds the global
    best cost after each iteration.
    """
    rng = np.random.default_rng(seed)
    sign = -1.0 if maximise else 1.0          # minimise sign * cost throughout

    def scored(w):
        c = sign * np.asarray(cost_fn(w), dtype=float)
        return np.where(np.isfinite(c), c, np.inf)

    x = project_to_simplex(rng.random((n_particles, n_dims)) * 0.5 + 0.1)
    v = rng.standard_normal((n_particles, n_dims)) * 0.01

    cost = scored(x)
    p_best, p_cost = x.copy(), cost.copy()
    g_idx = int(np.argmin(p_cost))
    g_best, g_cost = p_best[g_idx].copy(), p_cost[g_idx]
    history = [g_cost]

    for it in range(1, n_iterations):
        r1 = rng.random((n_particles, n_dims))
        r2 = rng.random((n_particles, n_dims))
        v = inertia * v + c_personal * r1 * (p_best - x) + c_global * r2 * (g_best - x)
        x = project_to_simplex(x + v)

        cost = scored(x)
        improved = cost < p_cost
        p_best[improved], p_cost[improved] = x[improved], cost[improved]
        g_idx = int(np.argmin(p_cost))
        if p_cost[g_idx] < g_cost:
            g_best, g_cost = p_best[g_idx].copy(), p_cost[g_idx]
        history.append(g_cost)

        if verbose and (it + 1) % 50 == 0:
            feasible = int(np.isfinite(cost).sum())
            print(f"  iter {it + 1:4d}/{n_iterations}  best {sign * g_cost:.6f}  feasible {feasible}/{n_particles}")

    if not np.isfinite(g_cost):
        raise RuntimeError(
            "PSO found no feasible light: every candidate was out of gamut or invalid. "
            "Check the reflectance inputs and LED spectra."
        )
    return g_best, sign * g_cost, sign * np.asarray(history)
