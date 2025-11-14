from __future__ import annotations
from typing import List, Tuple, Dict
import math
import random
from functools import lru_cache

def _normalize(items: List[float], C: float) -> List[float]:
    return [float(x) / float(C) for x in items]

def _restore_scale(bins: List[List[float]], C: float) -> List[List[float]]:
    return [[x * C for x in b] for b in bins]

def _validate_items(items: List[float], C: float):
    for x in items:
        if x < 0:
            raise ValueError("Item size must be non-negative")
        if x - C > 1e-12:
            raise ValueError(f"Item {x} exceeds bin capacity {C}")


def _group_large_items(items: List[float], eps: float) -> Tuple[List[float], List[float]]:
    large = sorted([w for w in items if w >= eps], reverse=True)
    small = [w for w in items if w < eps]
    return large, small

def _linear_group_round_with_map(desc_sizes: List[float], eps: float) -> Tuple[Dict[float, List[float]], List[float]]:
    """
    Returns (type_map, spill) where:
      - type_map: dict rounded_size -> list of ORIGINAL sizes rounded to that key.
      - spill: first group's ORIGINAL items (only when groups are meaningful).
    """
    n = len(desc_sizes)
    if n == 0:
        return {}, []

    G = min(max(1, int(math.ceil(1.0 / (eps*eps)))), n)
    gsize = int(math.ceil(n / G))
    groups = [desc_sizes[i:i+gsize] for i in range(0, n, gsize)]
    if gsize == 1 or len(groups) == 1:
        mp: Dict[float, List[float]] = {}
        for w in desc_sizes:
            mp.setdefault(w, []).append(w)
        return mp, []  

    spill = list(groups[0]) 
    mp: Dict[float, List[float]] = {}
    for grp in groups[1:]:
        if not grp:
            continue
        rounded_size = grp[0]
        mp.setdefault(rounded_size, []).extend(grp)  
    return mp, spill


def _types_and_multiplicities_from_map(type_map: Dict[float, List[float]]) -> Tuple[List[float], List[int]]:
    items = sorted(type_map.items(), key=lambda kv: -kv[0])
    types = [s for s,_ in items]
    mults = [len(v) for _,v in items]
    return types, mults


def _solve_rmp_full(types: List[float], mults: List[int], patterns: List[Tuple[int,...]], lp_solver_name: str = "PULP_CBC_CMD"):
    """
    Build & solve the Restricted Master Problem from scratch for the given pattern set.
    Returns (x_values, duals, obj_value, patterns) where:
      - x_values: list of variable values aligned with 'patterns'
      - duals: list of duals for each type-coverage constraint
      - obj_value: LP objective (lower bound on #bins for rounded large items)
    """
    import pulp
    m = len(types)
    P = list(range(len(patterns)))

    model = pulp.LpProblem("RMP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", P, lowBound=0)

    model += pulp.lpSum([x[p] for p in P])

    cons = []
    for i in range(m):
        cstr = pulp.lpSum(patterns[p][i] * x[p] for p in P) >= mults[i]
        cname = f"cov_{i}"
        model += cstr, cname
        cons.append(model.constraints[cname])

    solver = getattr(pulp, lp_solver_name)()
    model.solve(solver)

    x_val = [pulp.value(x[p]) for p in P]
    y = [cons[i].pi for i in range(m)]
    obj = pulp.value(model.objective)

    return x_val, y, obj, patterns

def _pricing_knapsack(types: List[float], duals: List[float], scale: int = 10000) -> Tuple[Tuple[int,...], float, float]:
    """
    Exact unbounded knapsack pricing:
      maximize sum_i duals[i]*c_i  s.t. sum_i types[i]*c_i <= 1, c_i integer >= 0.
    Returns (pattern_vector, profit, reduced_cost = profit - 1).
    We discretize capacity to 'scale' units (default 1e-4 precision).
    """
    m = len(types)
    cap = int(round(1.0 * scale))
    sizes = [int(round(t * scale)) for t in types]
    dp = [-1e100] * (cap + 1)
    dp[0] = 0.0
    prev = [(-1, -1)] * (cap + 1) 

    for i in range(m):
        w = sizes[i]
        v = float(duals[i] if duals[i] is not None else 0.0)
        for c in range(w, cap + 1):
            cand = dp[c - w] + v
            if cand > dp[c] + 1e-12:
                dp[c] = cand
                prev[c] = (i, c - w)

    best_c = max(range(cap + 1), key=lambda c: dp[c])
    profit = dp[best_c]

    pat = [0] * m
    c = best_c
    while c > 0 and prev[c][0] != -1:
        i, cprev = prev[c]
        pat[i] += 1
        c = cprev

    rc = profit - 1.0
    return tuple(pat), profit, rc

def _kk_cg_cover(types: List[float], mults: List[int], tol: float = 1e-9, max_iters: int = 10000, lp_solver_name: str = "PULP_CBC_CMD") -> List[Tuple[int,...]]:
    """
    Column generation for the rounded 'large' instance aggregated by type.
    Returns a multiset (list) of integer patterns that cover 'mults'.
    """
    m = len(types)
    if m == 0:
        return []

    patterns: List[Tuple[int,...]] = [tuple(1 if i == k else 0 for i in range(m)) for k in range(m)]

    it = 0
    while True:
        it += 1
        x_val, duals, obj, patterns = _solve_rmp_full(types, mults, patterns, lp_solver_name=lp_solver_name)
        pat, profit, rc = _pricing_knapsack(types, duals)
        if rc <= tol or it >= max_iters or sum(pat) == 0:
            break
        patterns.append(pat)

    x_val, duals, obj, patterns = _solve_rmp_full(types, mults, patterns, lp_solver_name=lp_solver_name)

    residual = list(mults)
    chosen: List[Tuple[int,...]] = []

    for p, x in enumerate(x_val):
        k = int(math.floor((x if x is not None else 0.0) + 1e-12))
        if k <= 0:
            continue
        chosen.extend([patterns[p]] * k)
        for i in range(m):
            residual[i] = max(0, residual[i] - k * patterns[p][i])

    guard = 0
    max_guard = 5 * m + 10 
    while any(residual):
        need_duals = [1.0 if residual[i] > 0 else 0.0 for i in range(m)]
        pat, profit, rc = _pricing_knapsack(types, need_duals)
        if sum(pat) == 0:
            i0 = max([i for i in range(m) if residual[i] > 0], key=lambda i: types[i])
            pat = tuple(1 if i == i0 else 0 for i in range(m))
        chosen.append(pat)
        for i in range(m):
            residual[i] = max(0, residual[i] - pat[i])
        guard += 1
        if guard > max_guard:
            for i in range(m):
                if residual[i] > 0:
                    chosen.extend([tuple(1 if j == i else 0 for j in range(m))] * residual[i])
                    residual[i] = 0
            break

    return chosen 

def _realize_large_bins_from_map(bins_cfg: List[Tuple[int,...]], types: List[float], type_map: Dict[float, List[float]]) -> List[List[float]]:
    stacks: Dict[float, List[float]] = {t: list(vals) for t, vals in type_map.items()}
    bins: List[List[float]] = []
    for cfg in bins_cfg:
        b: List[float] = []
        for t, cnt in zip(types, cfg):
            for _ in range(cnt):
                if not stacks[t]:
                    continue
                b.append(stacks[t].pop())
        if b:
            bins.append(b)
    for t in types:
        while stacks[t]:
            bins.append([stacks[t].pop()])
    return bins

def _pack_small_into_bins(bins: List[List[float]], small: List[float]) -> List[List[float]]:
    small = sorted(small, reverse=True)
    for w in small:
        placed = False
        for b in bins:
            if sum(b) + w <= 1.0 + 1e-12:
                b.append(w); placed = True; break
        if not placed:
            bins.append([w])
    return bins

def aptas_kk_like(items: List[float], C: float = 1.0, eps: float = 0.1, seed: int = 7, lp_solver_name: str = "PULP_CBC_CMD") -> Tuple[List[List[float]], Dict]:
    """
    KK Like APTAS for Bin Packing:
      - Linear grouping (round-up) with spill
      - Pattern LP with column generation to cover rounded large items
      - Realize with original sizes
      - Pack small items greedily
    Returns (bins_scaled_back, stats)
    """
    random.seed(seed)
    if not (0 < eps <= 0.5):
        raise ValueError("eps must be in (0, 0.5].")
    _validate_items(items, C)

    norm = _normalize(items, C)

    large_desc, small = _group_large_items(norm, eps)

    type_map, spill = _linear_group_round_with_map(large_desc, eps)
    types, mults = _types_and_multiplicities_from_map(type_map) if type_map else ([], [])

    bins_large_cfg: List[Tuple[int,...]] = _kk_cg_cover(types, mults, tol=1e-9, lp_solver_name=lp_solver_name) if types else []

    realized_large = _realize_large_bins_from_map(bins_large_cfg, types, type_map) if bins_large_cfg else []

    spill_bins = [[w] for w in spill] 

    all_bins = realized_large + spill_bins
    all_bins = _pack_small_into_bins(all_bins, small)
    out = _restore_scale(all_bins, C)
    for b in out:
        if sum(b) - C > 1e-8:
            raise AssertionError("Capacity violated.")
    flat = sorted([round(x, 12) for b in out for x in b])
    tgt  = sorted([round(x, 12) for x in items])
    if flat != tgt:
        raise AssertionError("Items lost/duplicated.")

    stats = {
        "n_items": len(items),
        "eps": eps,
        "n_large": len(large_desc),
        "n_small": len(small),
        "n_types": len(types),
        "n_patterns_selected": len(bins_large_cfg),
        "bins": len(out),
        "spill_count": len(spill),
        "total_volume": float(sum(items))/float(C) if C != 0 else 0.0,
        "lb_volume_ceiling": int(math.ceil(sum(items)/C - 1e-12)) if C != 0 else 0,
    }
    return out, stats

if __name__ == "__main__":
    items = [4,3,3,3,3,3]
    C = 10.0
    eps = 0.2  

    bins, stats = aptas_kk_like(items, C=C, eps=eps)
    print("Bins used:", len(bins))
    for b in bins:
        print(sorted([round(x,3) for x in b]), "sum=", round(sum(b),3))
    print("Stats:", stats)
