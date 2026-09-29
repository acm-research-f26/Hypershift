"""Independent four-point hyperbolicity calculations with inspectable witnesses.

Exact exhaustive evaluation for <=350 points; deterministic sampled lower bounds
otherwise. Diameter/2 is an upper bound. Equal bounds certify exactness. No
substitution of a sample maximum for an exact global maximum is permitted.
"""
from collections import deque
from itertools import combinations
from math import comb
import numpy as np


def witness(distances, vertices):
    a, b, c, d = map(int, vertices)
    pairs = {'ab': float(distances[a,b]), 'ac': float(distances[a,c]),
             'ad': float(distances[a,d]), 'bc': float(distances[b,c]),
             'bd': float(distances[b,d]), 'cd': float(distances[c,d])}
    sums = [pairs['ab'] + pairs['cd'], pairs['ac'] + pairs['bd'], pairs['ad'] + pairs['bc']]
    ordered = sorted(sums)
    return {'vertices': [a,b,c,d], 'distances': pairs, 'pair_sums': sums,
            'delta': (ordered[-1] - ordered[-2]) / 2}


def delta_metric(distances, exact_limit=350, samples=2000000, seed=20260928):
    d = np.asarray(distances, dtype=np.float64)
    n = len(d)
    if d.shape != (n,n) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Need a square, finite, nonnegative distance matrix.')
    if not np.allclose(d, d.T) or not np.allclose(d.diagonal(), 0):
        raise ValueError('Distances must be symmetric with a zero diagonal.')
    diameter = float(d.max())
    upper = diameter / 2
    if n < 4 or diameter == 0:
        return {'n': n, 'delta': 0.0, 'lower_bound': 0.0, 'upper_bound': 0.0,
                'relative_lower_bound': 0.0 if diameter else None,
                'diameter': diameter, 'exact': True, 'method': 'degenerate metric',
                'quadruples_evaluated': 0, 'possible_quadruples': comb(n,4) if n >= 4 else 0, 'witness': None}
    best, best_vertices, count = 0.0, (0,1,2,3), 0
    exact = n <= exact_limit
    def batch_value(a,b,c,e):
        first, second, third = d[a,b] + d[c,e], d[a,c] + d[b,e], d[a,e] + d[b,c]
        maximum = np.maximum(np.maximum(first, second), third)
        minimum = np.minimum(np.minimum(first, second), third)
        middle = first + second + third - maximum - minimum
        return np.maximum(0, (maximum - middle) / 2)
    if exact:
        # All distinct quadruples exactly once, with vectorized inner pair loop.
        for b in range(1, n-2):
            c, e = np.triu_indices(n-b-1, 1)
            c, e = c+b+1, e+b+1
            for a in range(b):
                values = batch_value(a,b,c,e)
                index = int(values.argmax())
                count += len(values)
                if values[index] > best:
                    best = float(values[index]); best_vertices = (a,b,c[index],e[index])
    else:
        rng = np.random.default_rng(seed)
        while count < samples:
            q = np.sort(rng.integers(n, size=(min(samples-count+2000, 200000),4)), axis=1)
            q = q[np.all(np.diff(q, axis=1) > 0, axis=1)][:samples-count]
            values = batch_value(*q.T)
            index = int(values.argmax())
            count += len(q)
            if values[index] > best:
                best = float(values[index]); best_vertices = tuple(q[index])
            if best >= upper - 1e-10:
                exact = True
                break
    method = 'exhaustive four-point' if n <= exact_limit else ('sample witness attains diameter/2 upper bound' if exact else 'sampled lower bound')
    if exact:
        upper = best
    return {'n': n, 'delta': best if exact else None, 'lower_bound': best,
            'upper_bound': upper, 'relative_lower_bound': 2*best/diameter,
            'relative_upper_bound': 2*upper/diameter, 'diameter': diameter,
            'exact': exact, 'method': method, 'seed': seed,
            'quadruples_evaluated': count, 'possible_quadruples': comb(n,4),
            'witness': witness(d, best_vertices)}


def euclidean_distances(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Features must be a finite matrix.')
    squares = np.sum(x*x, axis=1)
    d = np.sqrt(np.maximum(squares[:,None] + squares[None,:] - 2*x@x.T, 0))
    np.fill_diagonal(d, 0)
    return (d+d.T)/2


def adjacency_from_edges(n, edges):
    a = np.zeros((n,n), dtype=bool)
    e = np.asarray(edges, dtype=int).reshape(-1,2)
    if len(e):
        a[e[:,0],e[:,1]] = True
    a |= a.T.copy()
    np.fill_diagonal(a, False)
    return a


def neighborhood_groups(adjacency):
    return [set(np.flatnonzero(row)) | {i} for i,row in enumerate(adjacency)]


def merge_dice(groups, threshold, merge_similar=True):
    """Declared reconstruction: greedy union, extremal Dice first, stable ties.

    Not recovered from THINK code. Both inequality interpretations can be tried.
    """
    groups = [set(g) for g in groups]
    while len(groups) > 1:
        selected = None
        best = -1.0 if merge_similar else 2.0
        for i,j in combinations(range(len(groups)),2):
            score = 2*len(groups[i]&groups[j])/(len(groups[i])+len(groups[j]))
            eligible = score >= threshold if merge_similar else score < threshold
            improves = score > best if merge_similar else score < best
            if eligible and improves:
                selected, best = (i,j), score
        if selected is None:
            break
        i,j = selected
        groups[i] |= groups[j]
        del groups[j]
    return groups


def group_adjacency(n, groups, s=1):
    if s < 1:
        raise ValueError('s must be positive.')
    # Retain duplicate hyperedges unless caller explicitly deduplicates them.
    shared = np.zeros((n,n), dtype=np.int32)
    for group in groups:
        ids = np.array(sorted(group), dtype=int)
        shared[np.ix_(ids,ids)] += 1
    adjacency = shared >= s
    np.fill_diagonal(adjacency, False)
    return adjacency


def components(adjacency):
    remaining = set(range(len(adjacency)))
    result = []
    while remaining:
        seen, queue = {min(remaining)}, deque([min(remaining)])
        while queue:
            new = set(np.flatnonzero(adjacency[queue.popleft()])) - seen
            seen.update(new); queue.extend(new)
        remaining -= seen
        result.append(sorted(seen))
    return sorted(result, key=lambda c:(-len(c),c))


def shortest_paths(adjacency):
    """Exact unweighted BFS using bit sets for dense component frontiers."""
    n = len(adjacency)
    neighbors = [sum(1 << int(j) for j in np.flatnonzero(row)) for row in adjacency]
    distances = np.full((n,n), np.inf)
    for source in range(n):
        seen, frontier, level = 0, 1 << source, 0
        while frontier:
            next_frontier, bits = 0, frontier
            seen |= frontier
            while bits:
                bit = bits & -bits
                vertex = bit.bit_length()-1
                distances[source,vertex] = level
                next_frontier |= neighbors[vertex]
                bits ^= bit
            frontier = next_frontier & ~seen
            level += 1
    return distances


def graph_audit(adjacency, labels=None, **kwargs):
    groups = components(adjacency)
    results = []
    for nodes in groups:
        sub = adjacency[np.ix_(nodes,nodes)]
        result = delta_metric(shortest_paths(sub), **kwargs)
        if result['witness']:
            local = result['witness']['vertices']
            result['witness']['global_vertices'] = [nodes[i] for i in local]
            if labels is not None:
                result['witness']['labels'] = [str(labels[nodes[i]]) for i in local]
        results.append(result)
    return {'nodes': len(adjacency), 'undirected_edges': int(adjacency.sum()//2),
            'component_count': len(groups), 'component_sizes': [len(c) for c in groups],
            'global_delta': results[0]['delta'] if len(groups)==1 else None,
            'max_component_lower_bound': max(r['lower_bound'] for r in results),
            'max_component_upper_bound': max(r['upper_bound'] for r in results),
            'all_components_exact': all(r['exact'] for r in results),
            'components': results}
