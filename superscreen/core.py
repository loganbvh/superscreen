import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from matplotlib import path


def inpolygon(xq, yq, xp, yp):
    """Checks whether query points (xq, yq) lie in the polyon defined by points (xp, yp).
    Returns a boolean array of the same shape as xq and yq.
    """
    xq = np.asarray(xq)
    yq = np.asarray(yq)
    shape = xq.shape
    xq = np.asarray(xq).reshape(-1)
    yq = np.asarray(yq).reshape(-1)
    xp = np.asarray(xp).reshape(-1)
    yp = np.asarray(yp).reshape(-1)
    q = list(zip(xq, yq))
    p = path.Path(list(zip(xp, yp)))
    bool_array = p.contains_points(q).reshape(shape).squeeze()
    if len(bool_array.shape) == 0:
        bool_array = bool_array.item()
    return bool_array


def centroids(points, simplices):
    """Returns x,y coordinates for triangle centroids."""
    return np.array([np.sum(points[ix], axis=0) / 3 for ix in simplices])


def compute_adj(triangles: np.ndarray) -> np.ndarray:
    A = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    adj = csr_matrix((np.ones(A.shape[0]), (A[:, 0], A[:, 1])))
    adj = adj + adj.T
    adj = (adj > 0).astype(int)
    return np.asarray(adj.todense())


def area(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Calculate the area of each triangle.

    Args:
        points (np.ndarray): shape (n,2) array of x,y coordinates of nodes
        tris (np.ndarray) shape (m,3) array of triangles

    Returns:
        np.ndarray: shape (m,) array of triangle areas.
    """
    a = np.zeros(tris.shape[0])
    for i, e in enumerate(tris):
        xy = points[e]
        # s1 = xy[2, :] - xy[1, :]
        # s2 = xy[0, :] - xy[2, :]
        # s3 = xy[1, :] - xy[0, :]
        # which can be simplified to
        # s = xy[[2, 0, 1]] - xy[[1, 2, 0]]
        s = xy[[2, 0]] - xy[[1, 2]]
        # a should be positive if triangles are CCW arranged
        a[i] = la.det(s)
    return a * 0.5


def uniform_weights(adj: np.ndarray) -> np.ndarray:
    """Computes the weight matrix using uniform weighting."""
    weights = (adj != 0).astype(float)
    # normalize row-by-row
    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(weights, 1.0)
    return weights


def uniform_weights_for_loop(adj):
    n = adj.shape[0]
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):  # weights is symmetric, so only need
            # compute upper triangular part
            if adj[i, j]:  # Check if edge is in graph
                weights[i, j] = 1.0
    # Symmetrize weights
    weights = weights + weights.T
    # Normalize weights
    for i in range(n):
        weights[i, :] = weights[i, :] / sum(weights[i, :])
    for i in range(n):
        weights[i, i] = 1.0  # Fill in diagonal terms wii
    return weights


def q_matrix(points: np.ndarray) -> np.ndarray:
    """Computes the denominator matrix, Q."""
    distances = cdist(points, points)  # Euclidian distance between points
    q = np.zeros_like(distances)
    # Diagonals of distances are zero by definition, so q[i,i] will diverge
    nz = np.nonzero(distances)
    q[nz] = 1 / (4 * np.pi * distances[nz] ** 3)
    np.fill_diagonal(q, np.inf)  # diagonal elements diverge
    return q


def q_matrix_for_loop(points):
    n = points.shape[0]
    q = np.zeros((n, n))
    w_q = 1.0 / (4 * np.pi)  # Weight factor
    for i in range(n):
        for j in range(i + 1, n):  # Q is symmetric, so only need
            # compute upper triangular part
            n_rij = np.sqrt(
                (points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2
            )
            if n_rij == 0:
                raise ValueError("Coincident points in mesh!")
            q[i, j] = w_q / (n_rij ** 3)
    # Symmetrize q
    q = q + q.T
    for i in range(n):
        q[i, i] = np.inf  # Fill in diagonal terms qii = inf
    return q


def C_vector(points: np.ndarray) -> np.ndarray:
    """Computes the edge vector, C."""
    xmax = points[:, 0].max()
    xmin = points[:, 0].min()
    ymax = points[:, 1].max()
    ymin = points[:, 1].min()
    C = np.zeros(points.shape[0])
    for i, (x, y) in enumerate(points):
        with np.errstate(divide="ignore"):
            C_i = (
                np.sqrt((xmax - x) ** (-2) + (ymax - y) ** (-2))
                + np.sqrt((xmin - x) ** (-2) + (ymax - y) ** (-2))
                + np.sqrt((xmax - x) ** (-2) + (ymin - y) ** (-2))
                + np.sqrt((xmin - x) ** (-2) + (ymin - y) ** (-2))
            )
        C[i] = C_i
    C[np.isinf(C)] = 1e30
    return C / (4 * np.pi)


def laplacian(weights: np.ndarray, method: str = "uniform") -> np.ndarray:
    if method == "uniform":
        Del2 = np.triu(weights)
        Del2 = Del2 + Del2.T
        np.fill_diagonal(Del2, -1.0)
    else:
        raise NotImplementedError("Laplacian method {} not implemented.".format(method))
    return Del2


def laplacian_for_loop(weights, method="uniform"):
    if method != "uniform":
        raise NotImplementedError("Laplacian method {} not implemented.".format(method))
    n = weights.shape[0]
    Del2 = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):  # Del2 is symmetric, so only need
            # compute upper triangular part
            Del2[i, j] = weights[i, j]
    # Symmetrize Del2
    Del2 = Del2 + Del2.T
    for i in range(n):
        Del2[i, i] = -1.0  # Fill in diagonal terms Del2ii = -1
    return Del2


def Q_matrix(
    q: np.ndarray, C: np.ndarray, weights: np.ndarray, copy_q: bool = True
) -> np.ndarray:
    """Computes the kernel matrix, Q."""
    if copy_q:
        q = q.copy()
    # q[i, i] are np.inf, but Q[i, i] involves a sum over the off-diagonal
    # elements of q, so we can just set q[i, i] = 0 here.
    np.fill_diagonal(q, 0)
    Q = -np.triu(q)
    Q = Q + Q.T
    np.fill_diagonal(Q, (C + np.sum(q * weights, axis=1)) / np.diag(weights))
    return Q


def Q_matrix_for_loop(q, C, weights):
    n = weights.shape[0]
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):  # Q is symmetric, so we only need
            # compute upper triangular part
            Q[i, j] = -q[i, j]
    # Symmetrize Q
    Q = Q + Q.T
    Q_ij = 0
    # Fill in diagonal terms Qii
    for i in range(n):
        for k in range(n):
            if k != i:
                Q_ij = Q_ij + q[i, k] * weights[i, k]
        # weights[i,i] cancels out, so can be arbitrary
        Q[i, i] = (Q_ij + C[i]) / weights[i, i]
        Q_ij = 0
    return Q


# def brandt_2D(sparams, stype, vectorize=True):
#     if not isinstance(sparams, BrandtParams):
#         raise TypeError(
#             "Expected BrandtParams istance, but got {}.".format(type(sparams))
#         )
#     if stype not in ("coupled", "decoupled"):
#         raise ValueError("Expected 'coupled' or 'decoupled', but got {}.".format(stype))

#     # All length units in um
#     lambda_s = sparams.lambda_s  # London penetration depth
#     d = sparams.d  # Superconductor thickness
#     Ha_func = sparams.Ha  # Background magnetic field function
#     Lambda = lambda_s ** 2 / d  # Effective magnetic penetration depth
#     # Default parameters
#     Ir = 0  # No ring current
#     # Determine solution type (coupled: inductive coupling between layers)
#     if stype == "coupled":
#         # Import additional parameters from previous solution
#         Ha_0 = sparams.Ha_0
#         ph = sparams.ph  # Hole point coordinates
#         th = sparams.th  # Hole triangle indices
#         pr = sparams.pr  # Ring
#         tr = sparams.tr
#         pb = sparams.pb  # Bounding region
#         tb = sparams.tb
#         adj = sparams.adj
#         weights = sparams.weights
#         q = sparams.q
#         Q = sparams.Q
#         C = sparams.C
#         Del2 = sparams.Del2
#     elif stype == "decoupled":
#         # Get mesh from input argument
#         ph = sparams.ph
#         th = sparams.th
#         pr = sparams.pr
#         tr = sparams.tr
#         pb = sparams.pb
#         tb = sparams.tb
#         Ir = sparams.Ir
#         xmax = sparams.xmax
#         xmin = sparams.xmin
#         ymax = sparams.ymax
#         ymin = sparams.ymin

#     # Combine sub-meshes into global mesh:
#     # Delete coincident points in mesh and relabel points in triangles
#     nr = pr.shape[0]
#     # Offset hole indices by nr
#     th += nr
#     # Array of relabeled indices in hole mesh
#     ivh_r = np.arange(nr, nr + ph.shape[0])
#     ind_r = 0
#     # Check for coincidence in ring mesh
#     for i in range(nr):
#         # Hole mesh
#         ind = np.where((ph == pr[i, :]).all(axis=1))[0]
#         if ind.size > 1:
#             raise ValueError("Duplicate points on input mesh!")
#         elif ind.size != 0:
#             ind = ind[0]
#             ph = np.delete(ph, ind, 0)
#             ivh_r[ind + 1 + ind_r :] -= 1
#             ivh_r[ind + ind_r] = i
#             ind_r += 1
#             for ni in range(th.shape[0]):
#                 for nj in range(th.shape[1]):
#                     if th[ni, nj] == ind + nr:
#                         th[ni, nj] = i
#                     elif th[ni, nj] > ind + nr:
#                         th[ni, nj] -= 1
#     nh = ph.shape[0]
#     # Offset bounding indices by nr+nh
#     tb += nr + nh
#     # Array of relabeled indices in bounding mesh
#     ivb_r = np.arange(nr + nh, nr + nh + pb.shape[0])
#     ind_r = 0
#     for i in range(nr):
#         # Bounding region mesh
#         ind = np.where((pb == pr[i, :]).all(axis=1))[0]
#         if ind.size > 1:
#             raise ValueError("Duplicate points on input mesh!")
#         elif ind.size != 0:
#             ind = ind[0]
#             pb = np.delete(pb, ind, 0)
#             ivb_r[ind + 1 + ind_r :] -= 1
#             ivb_r[ind + ind_r] = i
#             for ni in range(tb.shape[0]):
#                 for nj in range(tb.shape[1]):
#                     if tb[ni, nj] == ind + nr + nh:
#                         tb[ni, nj] = i
#                     elif tb[ni, nj] > ind + nr + nh:
#                         tb[ni, nj] -= 1

#     nb = pb.shape[0]
#     iv = range(nr)  # Point indices of superconducting film
#     ivh = range(nr, nr + nh)  # Point indices of hole
#     p = np.concatenate((pr, ph, pb), axis=0)
#     t = np.concatenate((tr, th, tb), axis=0)
#     n_tr = tr.shape[0]
#     n_th = th.shape[0]
#     logging.info(t.shape[0])

#     if stype == "decoupled":
#         logging.info("Computing adjacency matrix adj... ")
#         adj = compute_adj(t)
#         logging.info("Done")
#     # Number of vertices in global mesh
#     n = adj.shape[0]
#     # -----------------------------------------------------------------------------
#     # Generate magnetic field data for layer
#     # -----------------------------------------------------------------------------
#     logging.info("Generating magnetic field data for layer... ")
#     if stype == "coupled":
#         Ha = Ha_func(p[:, 0], p[:, 1]) + Ha_0
#     else:
#         Ha = Ha_func(p[:, 0], p[:, 1])

#     if stype == "decoupled":

#         logging.info("Computing weight matrix weights... ")
#         if sparams.weight == "uniform":
#             if vectorize:
#                 weights = uniform_weights(adj)
#             else:
#                 weights = uniform_weights_for_loop(adj)

#         elif sparams.weight == "cotangent":
#             raise NotImplementedError('cotangent weighting not implemented.')
#             # weights = np.zeros((n, n))
#             # for i in range(n):
#             #     for j in range(i + 1, n):  # weights is symmetric, so only need
#             #         # compute upper triangular part
#             #         w_ij = 0
#             #         if adj[i, j]:  # Check if edge is in graph
#             #             # Identify alpha_ij and alpha_ji for adjacent polar vertices to edge ij
#             #             adj_v = np.nonzero(np.bitwise_and(adj[i, :], adj[j, :]))
#             #             if adj_v.size > 2:
#             #                 raise ValueError("Adjacency matrix definition error")
#             #             for k in range(adj_v.size):
#             #                 xi = p[i, :] - p[adj_v[k], :]  # Vector xi
#             #                 xj = p[j, :] - p[adj_v[k], :]  # Vector xj
#             #                 n_xi = la.norm(xi)
#             #                 n_xj = la.norm(xj)
#             #                 cos_a = 0.5 * (
#             #                     n_xi / n_xj
#             #                     + n_xj / n_xi
#             #                     - la.norm(xi - xj) ** 2 / (n_xi * n_xj)
#             #                 )
#             #                 if np.abs(cos_a) > 1:
#             #                     raise ValueError("Angle error")
#             #                 cot_a = cos_a / np.sqrt(1 - cos_a ** 2)
#             #                 w_ij = w_ij + 0.5 * cot_a
#             #             weights[i, j] = w_ij
#             # # Symmetrize weights
#             # weights = weights + weights.T
#             # # Check if all edge weights have been computed
#             # if not (np.array_equal(adj, weights != 0)):
#             #     raise ValueError("Weight matrix error")

#         logging.info("Computing denominator matrix Q... ")
#         if vectorize:
#             q = q_matrix(p)
#         else:
#             q = q_matrix_for_loop(p)

#         logging.info("Computing edge vector C... ")
#         if vectorize:
#             C = C_vector(p)
#         else:
#             C = C_vector(p)

#         logging.info("Computing Laplacian matrix Del2... ")
#         if vectorize:
#             Del2 = laplacian(weights)
#         else:
#             Del2 = laplacian_for_loop(weights)

#         logging.info("Computing dipole field kernel matrix Q... ")
#         if vectorize:
#             Q = Q_matrix(q, C, weights)
#         else:
#             Q = Q_matrix_for_loop(q, C, weights)

#     # -----------------------------------------------------------------------------
#     # Compute Ha_eff (sum only over points in hole)
#     # -----------------------------------------------------------------------------
#     ones = np.ones(nh)
#     Ha_eff = -Ir * (
#         (Q[:, ivh] * weights[:, ivh]) @ ones
#         - Lambda * (Del2[:, ivh] @ ones)
#     )

#     # -----------------------------------------------------------------------------
#     # Compute stream function gi using pivoted LU decomposition
#     # -----------------------------------------------------------------------------
#     # Restrict solution to subdomain of superconductor geometry
#     logging.info("Computing stream function gi... ")

#     # Form system A_Q.g = Ha (film only)
#     film = np.ix_(iv, iv)
#     A_Q = (
#         -Q[film] * weights[film]
#         + Lambda * Del2[film]
#     )
#     # PLU decomposition
#     P, L, U = la.lu(A_Q)
#     # Solve for g on film subdomain
#     g = np.zeros(n)
#     g[iv] = -la.solve(
#         U, la.solve(L, P @ (Ha[iv] + Ha_eff[iv]))
#     )
#     g[ivh] = Ir  # g = I in hole
#     logging.info("Done")
#     # Compute screened magnetic field Hz in layer
#     # Hz = Ha + np.matmul(np.multiply(Q, weights), gi)
#     Hz = Ha + (Q * weights) @ g
#     # Return computed solution and mesh parameters
#     sol = BrandtSolution()
#     sol.adj = adj
#     sol.weights = weights
#     sol.q = q
#     sol.Q = Q
#     sol.C = C
#     sol.Del2 = Del2
#     sol.g = g
#     sol.Hz = Hz
#     sol.nr = nr
#     sol.nh = nh
#     sol.nb = nb
#     sol.n_tr = n_tr
#     sol.n_th = n_th
#     sol.iv = iv
#     sol.ivh = ivh
#     sol.p = p
#     sol.t = t
#     sol.ivh_r = ivh_r
#     sol.ivb_r = ivb_r
#     return sol


# def brandt_layers(n_layers, sparams, stype):
#     sol = [0] * n_layers  # Solution information for each layer
#     Hzrt0 = [0] * n_layers  # Response fields without coupling
#     Hzrt0_sol = [0] * n_layers  # Response fields with coupling
#     if stype == "decoupled":
#         for i in range(n_layers):
#             logging.info("Solving for magnetic fields and currents on layer %d..." % i)
#             sol[i] = brandt_2D(sparams[i], "decoupled")
#     elif stype == "coupled":
#         # First solve each layer independently
#         for i in range(n_layers):
#             logging.info("Solving for magnetic fields and currents on layer %d..." % i)
#             sol[i] = brandt_2D(sparams[i], "decoupled")
#         logging.info("Propagating fields between layers... ")
#         for i in range(n_layers):
#             for j in range(n_layers):
#                 # Propagate fields between layers (layers j to each layer i, i != j)
#                 if j != i:
#                     # Find the center points of the triangles in plane j
#                     ntj = sol[j].t.shape[0]
#                     XYtj = np.zeros((ntj, 2))
#                     for nt in range(ntj):
#                         for nrow in range(3):
#                             XYtj[nt, 0] = XYtj[nt, 0] + sol[j].p[sol[j].t[nt, nrow], 0]
#                             XYtj[nt, 1] = XYtj[nt, 1] + sol[j].p[sol[j].t[nt, nrow], 1]
#                         XYtj[nt] = XYtj[nt] / 3
#                     triang = mtri.Triangulation(
#                         sol[j].p[:, 0], sol[j].p[:, 1], sol[j].t
#                     )
#                     gi_intrp = mtri.LinearTriInterpolator(triang, sol[j].Hz)
#                     # Interpolate gi in layer j to centers of mesh triangles
#                     gjt = gi_intrp(XYtj)
#                     # Calculate area of each triangle in plane j
#                     arj = area(sol[j].p, sol[j].t)
#                     # Compute Q(r, rp) for planes i and j
#                     z = sparams[j].z0 - sparams[i].z0
#                     npi = sparams[i].p.shape[0]
#                     ntj = sparams[j].t.shape[0]
#                     p1i = sparams[i].p
#                     p1j = sparams[j].p
#                     for ni in range(npi):
#                         qzsum0 = 0
#                         for nj in range(ntj):
#                             rho = np.sqrt(
#                                 (p1i[0, ni] - p1j[0, nj]) ** 2
#                                 + (p1i[1, ni] - p1j[1, nj]) ** 2
#                             )
#                             qzsum0 = qzsum0 + arj[nj] * gjt[nj] * (
#                                 2 * z ** 2 - rho ** 2
#                             ) / (4 * np.pi * np.power(z ** 2 + rho ** 2, 5.0 / 2))
#                         Hzrt0[i][ni] = Hzrt0[i][ni] + qzsum0
#         logging.info("Done")

#         # Solve again for magnetic fields and currents on each layer with inductive coupling
#         logging.info("Solving again with inductive coupling...")
#         for i in range(n_layers):
#             # Define solution parameters
#             sparams[i].Ha_0 = Hzrt0[i]
#             # Compute solution for layer and polygons
#             sol[i] = brandt_2D(sparams[i], "coupled")
#         logging.info("Done")
#     else:
#         raise ValueError("Invalid solution type")
#     logging.info("Done with solution")
#     return sol, Hzrt0, Hzrt0_sol
