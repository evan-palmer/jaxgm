from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax
from jax._src.typing import Array, ArrayLike

from jaxgm.linalg._matfuncs import schur

"""
None of this code is my own. This is a copy of the contribution in the following PR:
- https://github.com/jax-ml/jax/pull/25377
To support usage on GPUs, the schur decomposition has been manually implemented.
"""


@partial(jit, static_argnums=(1,))
def roots_legendre(n: int, max_n: int = 10) -> tuple[Array, Array]:
    """
    Compute roots and weights for Gauss-Legendre quadrature.
    JAX implementation of :func:`scipy.linalg.logm`.
    Args:
      n: quadrature order
      max_n: Maximum number of roots to compute
    Notes:
      Because of static array size requirements function in JAX takes `max_n` argument to determine size of output arrays.
      If `n < max_n` then entries for `n..max_n` will be zero and only positions below `n` will have valid nodes and weights.
      If `n >= max_n` then only first `max_n` roots and weights will be in the result.
      For inputs of 10000 and more, results may include NaNs because of precision errors.
    Examples:
      >>> nodes, weights = jax.scipy.special.roots_legendre(4, max_n=5)
      >>> with jnp.printoptions(precision=2, suppress=True):
      ...   print('nodes:', nodes)
      ...   print('weights:', weights)
      nodes: [-0.86 -0.34  0.34  0.86  0.  ]
      weights: [0.35 0.65 0.65 0.35 0.  ]
      To get all roots and weights use `max_n > n` and use only first `n` values
      >>> with jnp.printoptions(precision=2, suppress=True):
      ...   print('valid nodes:', nodes[:4])
      ...   print('valid weights:', weights[:4])
      valid nodes: [-0.86 -0.34  0.34  0.86]
      valid weights: [0.35 0.65 0.65 0.35]
    References:
      .. [1] W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery, Numerical Recipes
           in FORTRAN: The Art of Scientific Computing, 2nd ed., Cambridge University Press,
           London, 1992 (section 4.5 page 152)
    """
    eps = 3.0e-7
    m = (n + 1) // 2
    x = jnp.zeros(max_n, jnp.float32)
    w = jnp.zeros(max_n, jnp.float32)

    def perform_newton_method_refinement(data):
        z, z_old, pp = data
        p1 = 1.0
        p2 = 0.0
        p1, p2 = lax.fori_loop(
            0,
            n,
            lambda i, p: (((2.0 * i + 1.0) * z * p[0] - i * p[1]) / (i + 1.0), p[0]),
            (p1, p2),
        )
        pp = n * (z * p1 - p2) / (z * z - 1.0)
        z_old = z
        z = z_old - p1 / pp

        return z, z_old, pp

    def calculate_nth_root_and_weight(i, data):
        x, w = data
        z = jnp.cos(jnp.pi * (i + 1.0 - 0.25) / (n + 0.5))
        z1 = z + 1.0
        z, _, pp = lax.while_loop(
            lambda data: jnp.abs(data[0] - data[1]) > eps,
            perform_newton_method_refinement,
            (z, z1, 0.0),
        )
        x = lax.cond(i < max_n, lambda: x.at[i].set(-z), lambda: x)
        w = lax.cond(
            i < max_n, lambda: w.at[i].set(2.0 / ((1.0 - z * z) * pp * pp)), lambda: w
        )

        x = lax.cond(n - i - 1 < max_n, lambda: x.at[n - i - 1].set(z), lambda: x)
        w = lax.cond(n - i - 1 < max_n, lambda: w.at[n - i - 1].set(w[i]), lambda: w)
        return x, w

    x, w = lax.fori_loop(0, m, calculate_nth_root_and_weight, (x, w))

    return x, w


@jit
def _sqrtm_triu(T: Array) -> Array:
    """
    Implements Björck, Å., & Hammarling, S. (1983).
        "A Schur method for the square root of a matrix". Linear algebra and
        its applications", 52, 127-140.
    """
    diag = jnp.sqrt(jnp.diag(T))
    n = diag.size
    U = jnp.diag(diag)

    def i_loop(l, data):
        j, U = data
        i = j - 1 - l
        s = lax.fori_loop(i + 1, j, lambda k, val: val + U[i, k] * U[k, j], 0.0)
        value = jnp.where(T[i, j] == s, 0.0, (T[i, j] - s) / (diag[i] + diag[j]))
        return j, U.at[i, j].set(value)

    def j_loop(j, U):
        _, U = lax.fori_loop(0, j, i_loop, (j, U))
        return U

    U = lax.fori_loop(0, n, j_loop, U)
    return U


def _fractional_power_superdiag_entry(l1, l2, t12, p):
    """
    Compute a superdiagonal entry of a fractional matrix power.
    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798
    """

    def last_case():
        """Equation 5.5"""
        log_l1 = jnp.log(l1)
        log_l2 = jnp.log(l2)

        z = (l2 - l1) / (l2 + l1)

        if jnp.isrealobj(l1):
            #  for real values U is always 0 so return early to avoid casting to complex value
            return (
                t12
                * jnp.exp(p / 2 * (log_l2 + log_l1))
                * (2 * jnp.sinh(p * (jnp.arctanh(z))))
                / (l2 - l1)
            )

        # Equation 5.3
        U = jnp.ceil(((log_l2 - log_l1).imag - jnp.pi) / (2 * jnp.pi))

        return (
            t12
            * jnp.exp(p / 2 * (log_l2 + log_l1))
            * (2 * jnp.sinh(p * (jnp.arctanh(z) + jnp.pi * 1.0j * U.astype(l1.dtype))))
            / (l2 - l1)
        )

    case = lax.select(l1 == l2, 0, 2)
    case = lax.select(
        jnp.logical_or(jnp.abs(l1) < jnp.abs(l2) / 2, jnp.abs(l2) < jnp.abs(l1) / 2),
        1,
        case,
    )

    return lax.switch(
        case,
        [
            lambda: t12 * p * l1 ** (p - 1),
            lambda: t12 * ((l2**p) - (l1**p)) / (l2 - l1),
            last_case,
        ],
    )


def _logm_superdiag_entry(l1, l2, t12):
    """
    Compute a superdiagonal entry of a matrix logarithm.
    JAX implementation of the same function from scipy.
    This is like Eq. (11.28) in [1]_, except the determination of whether
    l1 and l2 are sufficiently far apart has been modified.
    References
    ----------
    .. [1] Nicholas J. Higham (2008)
            "Functions of Matrices: Theory and Computation"
            ISBN 978-0-898716-46-7
    """

    def last_case():
        z = (l2 - l1) / (l2 + l1)

        if jnp.isrealobj(l1):
            #  for real values U is always 0 so return early to avoid casting to complex value
            return t12 * 2.0 * jnp.arctanh(z) / (l2 - l1)

        log_diff = jnp.log(l2) - jnp.log(l1)
        U = jnp.ceil((log_diff.imag - jnp.pi) / (2 * jnp.pi))

        return (
            t12 * 2.0 * (jnp.arctanh(z) + jnp.pi * 1j * U.astype(l1.dtype)) / (l2 - l1)
        )

    case = lax.select(l1 == l2, 0, 2)
    case = lax.select(jnp.abs(l2 - l1) > jnp.abs(l1 + l2) / 2, 1, case)

    return lax.switch(
        case,
        [
            lambda: t12 / l1,
            lambda: t12 * (jnp.log(l2) - jnp.log(l1)) / (l2 - l1),
            last_case,
        ],
    )


def _briggs_helper_function(a: Array, k: int) -> Array:
    """
    Implements Awad H. Al-Mohy (2012) "A more accurate Briggs method for the logarithm",
           Numerical Algorithms, 59 : 393--402.
    """
    pi_half = jnp.pi / 2
    a_angle = jnp.angle(a)
    condition = a_angle >= pi_half
    a = jnp.where(condition, jnp.sqrt(a), a)
    z_0 = a - 1.0
    a = jnp.sqrt(a)
    r = a + 1.0

    def loop_body(i, x: tuple[Array, Array]) -> tuple[Array, Array]:
        a, r = x
        a = jnp.sqrt(a)
        r = r * (a + 1.0)
        return a, r

    a, r = lax.fori_loop(1, k - 1, loop_body, (a, r))
    #  one more loop step for k_hat=k
    _next_a, next_r = loop_body(0, (a, r))
    r = jnp.where(condition, r, next_r)
    r = z_0 / r
    return r


@partial(jit, static_argnames=("t", "itmax"))
def _onenormest(A: Array, key: ArrayLike, t: int = 2, itmax: int = 5) -> Array:
    """
    Estimate of the 1-norm of a matrix A.
    Implements Nicholas J. Higham and Francoise Tisseur (2000),
          "A Block Algorithm for Matrix 1-Norm Estimation,
          with an Application to 1-Norm Pseudospectra."
          SIAM J. Matrix Anal. Appl. Vol. 21, No. 4, pp. 1185-1201.
    """
    n = A.shape[-1]
    if t >= n:
        # if t is greater than number of columns it is faster to just compute exact value
        # we also avoid getting stuck in an infinite loop when generating vectors that are not parallel in algorithm
        return jnp.linalg.norm(A, 1, axis=(-2, -1))
    ind_hist = jnp.ones(t * itmax, dtype=jnp.int32) * -1
    est_old = jnp.zeros(dtype=A.dtype, shape=[]).real
    idx_size = min(n, t * itmax + t)
    ind = jnp.zeros((idx_size,), dtype=jnp.int32)
    S = jnp.zeros((n, t), dtype=A.dtype)
    k = jnp.array(1, dtype=jnp.int32)
    itmax_ = jnp.array(itmax, dtype=jnp.int32)

    #  initialize starting matrix X with columns of unit 1-norm
    #  choice of columns is explained in scipy/sparse/linalg/_onenormest.py
    X = jnp.ones((n, t), dtype=A.dtype)

    def needs_resampling(data, i: int):
        X, key = data
        return (X[:, :i].T @ X[:, i] == n).any()

    def resample(data, i: int):
        X, key = data
        key, subkey = jax.random.split(key)
        rand_val = (
            jax.random.randint(subkey, shape=X.shape[0], minval=0, maxval=2) * 2 - 1
        )
        X = X.at[:, i].set(rand_val.astype(X.dtype))
        return X, key

    if t > 1:
        for i in range(1, t):
            key, subkey = jax.random.split(key)
            rand_val = (
                jax.random.randint(subkey, shape=[X.shape[0]], minval=0, maxval=2) * 2
                - 1
            )
            X = X.at[:, i].set(rand_val.astype(X.dtype))
        for i in range(t):
            #  resample if column of X is parallel to a previous column
            #  Parrarel vectors will are equal or opposite in this case so their dot product is n
            X, key = jax.lax.while_loop(
                partial(needs_resampling, i=i), partial(resample, i=i), (X, key)
            )

    X /= n

    def needs_resampling2(data, i: int):
        S, S_old, key = data
        cond_1 = (S[:, :i].T @ S[:, i] == n).any()
        cond_2 = (S_old.T @ S[:, i] == n).any()
        return jnp.logical_or(cond_1, cond_2)

    def resample2(data, i: int):
        S, S_old, key = data
        key, subkey = jax.random.split(key)
        rand_val = (
            jax.random.randint(subkey, shape=S.shape[0], minval=0, maxval=2) * 2 - 1
        )
        S = S.at[:, i].set(rand_val.astype(S.dtype))
        return S, S_old, key

    def main_loop_body(x):
        #  In this function instead of using break or goto we set k to itmax
        #  This way loop will terminate after ending current iteration
        A, X, S, ind, ind_hist, est_old, key, k = x

        Y = A @ X
        summed_abs_cols = jnp.abs(Y).sum(0)
        est = jnp.max(summed_abs_cols)
        ind_j = jnp.argmax(summed_abs_cols)
        ind_best = ind[ind_j]

        est, k = jax.lax.cond(
            jnp.logical_and(k >= 2, est <= est_old),
            (lambda: (est_old, itmax_)),
            (lambda: (est, k)),
        )

        est_old = est
        S_old = S

        S = (Y + (Y == 0).astype(Y.dtype)) / jnp.abs(Y).astype(Y.dtype)

        # if all vectors in S are parallel to vector in S_old finish iterating
        k = jax.lax.cond((S.T @ S_old == n).all(), lambda: itmax_, lambda: k)

        if t > 1:
            # Ensure that no column of S is parallel to another column of S
            # or to a column of S_old by replacing columns of S by rand{−1, 1}
            for i in range(t):
                S, S_old, key = jax.lax.while_loop(
                    partial(needs_resampling2, i=i),
                    partial(resample2, i=i),
                    (S, S_old, key),
                )

        Z = A.T @ S
        h = jnp.abs(Z).max(1)
        k = jax.lax.cond(
            jnp.logical_and(k >= 2, (jnp.max(h) == h[ind_best]).all()),
            lambda old_k: itmax_,
            lambda old_k: old_k,
            k,
        )
        ind = jnp.argsort(h, descending=True)[: t + len(ind_hist)].astype(
            ind_hist.dtype
        )
        if t > 1:
            k = jax.lax.cond(
                jnp.isin(ind[:t], ind_hist).all(), lambda: itmax_, lambda: k
            )
            # put not seen indices first
            seen = jnp.isin(ind, ind_hist)
            idx = jnp.argsort(seen, stable=True)
            ind = ind[idx]

        elementary_vectors = jax.nn.one_hot(ind, n).T
        X = elementary_vectors[:, :t].astype(A.dtype)
        new_ind = ind[:t].copy()
        ind_hist = jax.lax.dynamic_update_slice(ind_hist, new_ind, (k * t,))
        k += 1
        return A, X, S, ind, ind_hist, est_old, key, k

    def main_loop_cond(x):
        A, X, S, ind, ind_hist, est_old, key, k = x
        return k < itmax

    A, X, S, ind, ind_hist, est, key, k = jax.lax.while_loop(
        main_loop_cond, main_loop_body, (A, X, S, ind, ind_hist, est_old, key, k)
    )

    return est


@jit
def _inverse_squaring(T_0: Array, theta: tuple[float], key: ArrayLike):
    """
    Implements lines 3--34 of algoritm 4.1 in Awad H. Al-Mohy and Nicholas J. Higham (2012)
             "Improved Inverse Scaling and Squaring Algorithms
             for the Matrix Logarithm."
    """

    def normest(T: Array, p: int, key: ArrayLike):
        T = jnp.linalg.matrix_power(T - jnp.eye(T.shape[0], dtype=T.dtype), p)
        return _onenormest(T, key)

    T = T_0
    diag = jnp.diag(T)
    s_0 = 0

    def cond(x):
        diag, s_0 = x
        return jnp.max(jnp.abs(diag - 1)) > theta[7]

    def body(x):
        diag, s_0 = x
        diag = jnp.sqrt(diag)
        s_0 += 1
        return diag, s_0

    diag, s_0 = jax.lax.while_loop(cond, body, (diag, s_0))

    T = jax.lax.fori_loop(0, s_0, lambda i, T: _sqrtm_triu(T), T)

    s = s_0
    k = 0
    d_2 = normest(T, 2, key) ** (1 / 2)
    d_3 = normest(T, 3, key) ** (1 / 3)
    a_2 = jnp.maximum(d_2, d_3)
    m = 0
    for i in (1, 2):
        m = jax.lax.cond(a_2 < theta[i], lambda m: i, lambda m: m, m)

    def main_loop_cond(x):
        T, s, m = x
        return m == 0

    def main_loop_body(x):
        T, s, m = x
        nonlocal d_3
        nonlocal k
        d_3 = jax.lax.cond(s > s_0, lambda: normest(T, 3, key) ** (1 / 3), lambda: d_3)
        d_4 = normest(T, 4, key) ** (1 / 4)
        a_3 = jnp.maximum(d_3, d_4)

        def fun(m, k):
            # 18 to 27
            ind = jnp.arange(3, 8)
            for i, idx in enumerate(ind):
                ind = jax.lax.select(a_3 <= theta[idx], ind, ind.at[i].set(8))

            j_1 = jnp.min(ind)
            m = jax.lax.select(j_1 <= 6, j_1, m)
            should_continue = jnp.logical_and(
                jnp.logical_and(a_3 / 2 <= theta[5], k < 2), m == 0
            )
            k = jax.lax.select(should_continue, k + 1, k)
            return m, k, should_continue

        # 17
        m, k, should_continue = jax.lax.cond(
            a_3 < theta[7], fun, lambda m, k: (m, k, False), m, k
        )
        # should continue is goto 33 from original algorithm
        d_5 = normest(T, 5, key) ** (1 / 5)
        a_4 = jnp.maximum(d_4, d_5)
        eta = jnp.minimum(a_3, a_4)
        for i in (6, 7):
            condition = jnp.logical_and(m == 0, eta < theta[i])
            condition = jnp.logical_and(condition, ~should_continue)
            m = jax.lax.select(condition, i, m)

        T, s = jax.lax.cond(m == 0, lambda: (_sqrtm_triu(T), s + 1), lambda: (T, s))
        return T, s, m

    T, s, m = jax.lax.while_loop(main_loop_cond, main_loop_body, (T, s, m))

    #  R = (T - I), but we compute it with briggs algorithm to avoid cancellation on diagonal
    R = T
    a = jnp.diag(T_0)
    R = jnp.fill_diagonal(R, _briggs_helper_function(a, s), inplace=False)

    # replace superdiagonal
    p = jnp.exp2(-s)

    def replace_superdiag_fn(i, A):
        l1 = T_0[i, i]
        l2 = T_0[i + 1, i + 1]
        t12 = T_0[i, i + 1]
        return A.at[i, i + 1].set(_fractional_power_superdiag_entry(l1, l2, t12, p))

    R = lax.fori_loop(0, T.shape[-1] - 1, replace_superdiag_fn, R)

    has_principal_branch = jnp.logical_or(diag.real > 0, diag.imag != 0).all()
    R = lax.select(
        has_principal_branch, R, T - jnp.identity(T.shape[-1]).astype(T.dtype)
    )

    return R, s, m


@jit
def _logm_triu(T: Array, key: ArrayLike) -> Array:
    """
    Implements Awad H. Al-Mohy and Nicholas J. Higham (2012) "Improved Inverse Scaling and Squaring Algorithms for the Matrix Logarithm."
    """
    n = T.shape[-1]
    diag = jnp.diag(T)
    T_0 = T
    #  Bounds defined in table 2.1 from Awad H. et al.
    #  first entry set to NaN to offset indexes by 1 because they start from 1 in the paper
    theta_m = jnp.array(
        [
            float("nan"),
            1.59e-5,
            2.31e-3,
            1.94e-2,
            6.21e-2,
            1.28e-1,
            2.06e-1,
            2.88e-1,
            3.67e-1,
            4.39e-1,
            5.03e-1,
            5.60e-1,
            6.09e-1,
            6.52e-1,
            6.89e-1,
            7.21e-1,
            7.49e-1,
        ],
        dtype=T.dtype,
    ).real
    R, s, m = _inverse_squaring(T_0, theta_m, key=key)

    # line 36 of algorithm 4.1
    # evaluate U = 2^s * r_m(T-I)
    nodes, weights = roots_legendre(m, max_n=7)
    # move nodes and weights from range [-1,1] to [0,1]
    nodes = ((nodes + 1.0) / 2.0).astype(R.dtype)
    weights = (weights / 2.0).astype(R.dtype)
    identity = jnp.identity(n, dtype=T.dtype)
    U = jnp.zeros_like(R)
    U = lax.fori_loop(
        0,
        m,
        lambda i, U: U
        + jax.scipy.linalg.solve_triangular(identity + R * nodes[i], R * weights[i]),
        U,
    )
    U = U * jnp.exp2(s)

    has_principal_branch = jnp.logical_or(diag.real > 0, diag.imag != 0).all()
    # replace diagonal
    U2 = jnp.fill_diagonal(U, jnp.log(diag), inplace=False)

    # replace superdiagonal
    def replace_superdiag_fn(i, A):
        l1 = T_0[i, i]
        l2 = T_0[i + 1, i + 1]
        t12 = T_0[i, i + 1]
        return A.at[i, i + 1].set(_logm_superdiag_entry(l1, l2, t12))

    U2 = lax.fori_loop(0, n - 1, replace_superdiag_fn, U)

    U = lax.select(has_principal_branch, U2, U)

    return U


@jit
def logm(A: ArrayLike, key: ArrayLike) -> Array:
    """Compute matrix logarithm
    JAX implementation of :func:`scipy.linalg.logm`.
    Args:
      A: array of shape ``(N, N)``
    Returns:
      An array of shape ``(N, N)`` containing the matrix logarithm of ``A``.
    Examples:
      >>> A = jnp.array([[1., 2., 3.],
      ...                [2., 4., 2.],
      ...                [3., 2., 1.]])
      >>> log_a = jax.scipy.linalg.logm(a, key=jax.random.key(0))
      >>> with jnp.printoptions(precision=2, suppress=True):
      ...   print(log_a)
      [[0.87+1.57j 0.62+0.j   0.17-1.57j]
       [0.62-0.j   1.04-0.j   0.62+0.j  ]
       [0.17-1.57j 0.62-0.j   0.87+1.57j]]
       By definition, matrix multiplication is inverse of exponentiation:
      >>> jnp.allclose(a, jax.scipy.linalg.expm(log_a))
      Array(True, dtype=bool)
    Notes:
      This uses the inverse scaling-and-squaring approximation method.
    References:
      .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
             "Improved Inverse Scaling and Squaring Algorithms for the Matrix Logarithm."
             SIAM Journal on Scientific Computing, 34 (4). C152-C169.
             ISSN 1095-7197
    """
    if not jnp.isrealobj(A):
        raise NotImplementedError("Complex matrices are not supported.")

    def perform_real_logm(T, Z):
        logm_T = _logm_triu(T, key=key)
        complex_dtype = jnp.complex64 if T.dtype == jnp.float32 else jnp.complex_
        return logm_T.astype(complex_dtype), Z.astype(complex_dtype)

    def perform_complex_logm(T, Z):
        T, Z = jax.scipy.linalg.rsf2csf(T, Z)
        logm_T = _logm_triu(T, key=key)
        return logm_T, Z

    T, Z = schur(A)
    keep_it_real = jnp.logical_and(
        jnp.array_equal(T, jnp.triu(T)), jnp.min(jnp.diag(T)) >= 0
    )
    logm_T, Z = lax.cond(keep_it_real, perform_real_logm, perform_complex_logm, T, Z)

    return jnp.matmul(
        jnp.matmul(Z, logm_T, precision=lax.Precision.HIGHEST),
        jnp.conj(Z.T),
        precision=lax.Precision.HIGHEST,
    )
