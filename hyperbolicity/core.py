from __future__ import annotations
from collections import deque 
from itertools import combinations

import numpy as np

 # s: Number of hyperedges two stocks must share to be adjacent
def s_walk_adjacency(incidence: np.ndarray, s: int = 1) -> np.ndarray:
    memberships = incidence.astype(np.int64, copy=False)
    shared_counts = memberships @ memberships.T  # $HH^\top \text{ counts shared hyperedges}$
    adjacency = shared_counts >= s # returns boolean 1 if two nodes adjacent, otherwise 0 as a matrix
    np.fill_diagonal(adjacency, False) # Avoid self loops
    return adjacency


def shortest_path_distances(adjacency: np.ndarray) -> np.ndarray:
    # BFS to compute distances to each node (how many hops away)
    if adjacency.ndim != 2 or 0 in adjacency.shape:
        raise ValueError("Adjacency matrix must be nonempty and two-dimensional.")

    n = adjacency.shape[0] # Number of nodes
    distances = np.full((n,n), np.inf, dtype=float)  # $\text{Set unreached distances to }\infty$
    for source in range(n):
        distances[source, source] = 0.0
        queue = deque([source])

        while queue:
            current = queue.popleft()
             # Takes the current node and returns indices where values are nonzero (flattened), i.e. there is a neighboring node
            for neighbor_index in np.flatnonzero(adjacency[current]):
                neighbor = int(neighbor_index)
                if np.isinf(distances[source, neighbor]):
                    distances[source, neighbor] = (distances[source, current] + 1.0)
                    queue.append(neighbor)

    if np.isinf(distances).any():
        raise ValueError("Graph is disconnected.")
    return distances

# $\text{THINK Eq. (1): }(y,z)_x = \frac{1}{2}(d(x,y) + d(x,z) - d(y,z))$
# $\text{THINK Eq. (2): }(x,z)_w \geq \min\{(x,y)_w, (y,z)_w\} - \delta$
#
# $\text{Substitute Eq. (1) into Eq. (2), multiply by }2\text{, and rearrange}$
# $d(w,y) + d(x,z) \leq \max\{d(w,x) + d(y,z), d(w,z) + d(x,y)\} + 2\delta$
#
# $\text{Let the three opposite-pair sums be}$
# $S_1 = d(w,x) + d(y,z)$
# $S_2 = d(w,y) + d(x,z)$
# $S_3 = d(w,z) + d(x,y)$
# $\text{For every ordering, the largest sum must satisfy }S_{\max} \leq S_{\mathrm{middle}} + 2\delta$
# $\text{Therefore, the smallest valid value is }\delta = \frac{S_{\max} - S_{\mathrm{middle}}}{2}$
def four_point_delta(distances: np.ndarray) -> float:
    # Given 4 points, measure how far the geometry is from behaving like a tree. 
    pairing_sums = [
        distances[0, 1] + distances[2, 3],
        distances[0, 2] + distances[1, 3],
        distances[0, 3] + distances[1, 2],
    ]

    pairing_sums.sort()

    middle = pairing_sums[1]
    largest = pairing_sums[2]

    return float((largest - middle) / 2.0)

# Analytical, however impractical for any reasonably sized stock universe.
# For 250 stocks this is nC4 
# i.e. 158,882,750 quadruples that need to be checked. 
def exact_delta_details(distances: np.ndarray, *, block_size: int = 16) -> dict:
    """Enumerate distinct quadruples in bounded NumPy blocks; retain a witness.

    Fixing the second index lets every block reuse the same upper-triangle
    indices. No O(n**4) array is allocated. This is exact enumeration, not a
    claim to implement a subquartic max-min matrix multiplication algorithm.
    """
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or not len(d) or d.shape[0] != d.shape[1] or not np.isfinite(d).all():
        raise ValueError("Exact delta requires a finite nonempty square distance matrix")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    n, best, witness = len(d), 0.0, None
    for j in range(1, n - 2):
        k, l = np.triu_indices(n - j - 1, 1)
        k, l = k + j + 1, l + j + 1
        for start in range(0, j, block_size):
            i = np.arange(start, min(j, start + block_size))[:, None]
            a = d[i, j] + d[k, l][None, :]
            b = d[i, k[None, :]] + d[j, l][None, :]
            c = d[i, l[None, :]] + d[j, k][None, :]
            maximum = np.maximum(np.maximum(a, b), c)
            middle = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
            values = (maximum - middle) * 0.5
            flat = int(values.argmax())
            value = float(values.flat[flat])
            if value > best:
                row, col = np.unravel_index(flat, values.shape)
                best, witness = value, (int(i[row, 0]), j, int(k[col]), int(l[col]))
    return {"delta": best, "lower_bound": best, "upper_bound": best,
            "method": "exact_chunked", "witness": witness,
            "quadruples": int(n * (n - 1) * (n - 2) * (n - 3) // 24) if n >= 4 else 0}


def exact_delta(distances: np.ndarray) -> float:
    return exact_delta_details(distances)["delta"]


def basepoint_bounds(distances: np.ndarray, *, n_basepoints: int = 4, seed: int = 0) -> dict:
    """Certified bounds max(delta_r) <= delta <= min(2*delta_r, diameter/2)."""
    d = np.asarray(distances, dtype=np.float64)
    validate_distances(d)
    if n_basepoints < 1:
        raise ValueError("At least one basepoint is required")
    rng = np.random.default_rng(seed)
    roots = rng.choice(len(d), min(n_basepoints, len(d)), replace=False)
    lower, upper, values = 0.0, float(d.max()) / 2, []
    for root in roots:
        product = (d[root, :, None] + d[root, None, :] - d) / 2
        composed = np.zeros_like(product)
        for k in range(len(d)):
            np.maximum(composed, np.minimum(product[:, k, None], product[k, None, :]), out=composed)
        value = max(0.0, float(np.max(composed - product)))
        lower, upper = max(lower, value), min(upper, 2 * value)
        values.append(value)
    return {"delta": None, "lower_bound": lower, "upper_bound": max(lower, upper),
            "method": "certified_basepoints", "basepoints": roots.tolist(),
            "basepoint_deltas": values, "seed": seed}

# Points is a matrix of feature vectors
def euclidean_distances(points: np.ndarray) -> np.ndarray:
    
    points_float = points.astype(float, copy=False)

    # $\text{Return one squared norm per row: }\|x_i\|^2 = \sum_j x_{ij}^2$
    # Example:
    # $\operatorname{einsum}(\mathtt{ij,ij\to i}, X, X) = \begin{bmatrix}x_1^\top x_1\\x_2^\top x_2\end{bmatrix}$
    
    # $\text{For }X=\begin{bmatrix}1&2\\3&4\end{bmatrix}\text{, the rows are }x_1^\top=\begin{bmatrix}1&2\end{bmatrix}\text{ and }x_2^\top=\begin{bmatrix}3&4\end{bmatrix}$

    # $\text{Remove this and put the matrix here}$

    # $= \begin{bmatrix}1^2+2^2\\3^2+4^2\end{bmatrix} = \begin{bmatrix}5\\25\end{bmatrix}$
    
    squared_norms = np.einsum(
        "ij,ij->i",
        points_float,
        points_float,
    )

    dot_products = points_float @ points_float.T

    # $\text{Expand the squared distance as an inner product}$
    # $\|x-y\|^2 = \langle x-y,x-y\rangle$
    # $= \langle x,x\rangle - \langle x,y\rangle - \langle y,x\rangle + \langle y,y\rangle$
    # $= \|x\|^2 + \|y\|^2 - 2\langle x,y\rangle$
    squared_distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * dot_products
    )

    squared_distances = np.maximum(squared_distances, 0.0)
    distances = np.sqrt(squared_distances)

    # Distance from a point to itself is 0
    np.fill_diagonal(distances, 0.0)
    
    # $\text{Return }D\in\mathbb{R}^{N\times N}\text{ where }D_{ij}=\|x_i-x_j\|$
    return distances


# $\text{Interpret }\delta_{hg}\text{ and }\delta_{\mathrm{rel}}\text{ qualitatively}$
# $\delta_{hg}\text{ low},\ \delta_{\mathrm{rel}}\text{ low}$
# $\text{Both structural and temporal-feature metrics are tree-like}$
# $\text{This does not by itself prove that they align or improve prediction}$
#
# $\delta_{hg}\text{ low},\ \delta_{\mathrm{rel}}\text{ high}$
# $\text{The hypergraph is tree-like, but the temporal-feature geometry is not}$
# $\text{The hypergraph construction may be imposing an unsupported hierarchy}$
#
# $\delta_{hg}\text{ high},\ \delta_{\mathrm{rel}}\text{ low}$
# $\text{The temporal-feature geometry is tree-like, but the hypergraph is not}$
# $\text{The chosen hyperedges may be failing to capture the feature hierarchy}$
#
# $\delta_{hg}\text{ high},\ \delta_{\mathrm{rel}}\text{ high}$
# $\text{Neither metric is strongly tree-like under these diagnostics}$
# $\text{Hyperbolic modeling is then less motivated by these scores alone}$
#
# $\delta_{hg}\text{ is raw and }\delta_{\mathrm{rel}}\text{ is dimensionless}$
# $\text{Judge low and high within each metric; do not compare their values directly}$
def relative_delta(delta: float, distances: np.ndarray) -> float | None:
    # $\operatorname{diam}(X)=\max_{i,j}D_{ij}$
    diameter = float(np.max(distances))

    # $\operatorname{diam}(X)=0\Longrightarrow\delta_{\mathrm{rel}}\text{ is undefined}$
    if diameter == 0.0:
        return None

    # $\delta_{\mathrm{rel}}=\frac{2\delta}{\operatorname{diam}(X)}$
    return float((2.0 * delta) / diameter)

def validate_distances(distances: np.ndarray, *, atol: float = 1e-10) -> None:
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative.")

    # $D\in\mathbb{R}^{N\times N},\quad N>0$
    if (
        distances.ndim != 2
        or distances.shape[0] == 0
        or distances.shape[0] != distances.shape[1]
    ):
        raise ValueError("Distances must be a nonempty square matrix.")

    # $d(i,j)\in\mathbb{R}$
    if not np.isfinite(distances).all():
        raise ValueError("Distances must contain only finite values.")

    # $d(i,j)\geq0$
    if np.any(distances < -atol):
        raise ValueError("Distances must be nonnegative.")

    # $d(i,j)=d(j,i)$
    if not np.allclose(
        distances,
        distances.T,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("Distance matrix must be symmetric.")

    # $d(i,i)=0$
    if not np.allclose(
        np.diag(distances),
        0.0,
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("Distance matrix diagonal must be zero.")

    # $d(i,j)\leq d(i,k)+d(k,j)$
    for k in range(distances.shape[0]):
        distance_to_k = distances[:, k][:, None]
        distance_from_k = distances[k, :][None, :]
        distance_via_k = distance_to_k + distance_from_k

        if np.any(distances > distance_via_k + atol):
            raise ValueError("Distance matrix violates the triangle inequality.")

    return None

def sampled_delta(distances: np.ndarray, *, n_samples: int = 10_000, seed: int = 0) -> float:
    n = distances.shape[0]

    if n < 4:
        raise ValueError("At least four points are required.")

    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples <= 0
    ):
        raise ValueError("n_samples must be a positive integer.")

    rng = np.random.default_rng(seed)
    largest_observed_delta = 0.0

    for _ in range(n_samples):
        indices = rng.choice(n, size=4, replace=False)
        quadruple_distances = distances[np.ix_(indices, indices)]
        quadruple_delta = four_point_delta(quadruple_distances)

        largest_observed_delta = max(
            largest_observed_delta,
            quadruple_delta,
        )

    # $\widehat{\delta}=\max_{q\in Q_s}\delta(q)\leq\delta^*$
    return float(largest_observed_delta)

# Also going to look into 
# Certified fixed-basepoint approximation - Fournier, Ismail, and Vigneron
# Duan (1 + eps), (2 + eps) approximations
