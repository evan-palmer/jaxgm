from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import jit
from jax.typing import DTypeLike
from jaxtyping import Array, Num, jaxtyped

from jaxgm.lie_group import to_parameters
from jaxgm.linalg._norm import damped_norm, frobenius_norm
from jaxgm.linalg._vecfuncs import skew3, vex3
from jaxgm.rotation import rotation_angle


@partial(jit, static_argnames=("tol",))
@jaxtyped(typechecker=beartype)
def sqrtm_pd(A: Num[Array, "n n"], tol: float = 1e-6) -> Num[Array, "n n"]:
    """Square root of a positive definite matrix using Denman-Beaver's scaled iteration.

    Parameters
    ----------
    A : Num[Array, "n n"]
        The (positive-definite) matrix to compute the square root of.
    p : int
        The number of iterations to perform.

    Returns
    -------
    Num[Array, "n n"]
        The matrix square root of `A`.
    """
    I = jnp.eye(A.shape[-1], dtype=A.dtype)  # noqa: E741
    X = A
    Y = I

    n = X.shape[-1]

    def make_step(args):
        X, Y = args

        # Use determinant-based scaling
        scale = jnp.abs(1 / (jnp.linalg.det(X) * jnp.linalg.det(Y))) ** (1 / n)

        # Denman-Beavers iteration
        X_n = 0.5 * (scale * X + (1 / scale) * jnp.linalg.inv(Y))
        Y = 0.5 * (scale * Y + (1 / scale) * jnp.linalg.inv(X))
        X = X_n

        return X, Y

    def cond(args):
        X, _ = args
        return frobenius_norm(jnp.linalg.matrix_power(X, 2) - A) > tol

    X, Y = jax.lax.while_loop(cond, make_step, (X, Y))

    return X


@jaxtyped(typechecker=beartype)
def is_pd(A: Num[Array, "n n"]) -> DTypeLike:
    """Check if a matrix is positive definite.

    Parameters
    ----------
    A : Num[Array, "n n"]
        The matrix to check.

    Returns
    -------
    bool
        True if the matrix is positive definite, False otherwise.
    """
    return jnp.all(jnp.linalg.eigvals(A) > 0)


@jaxtyped(typechecker=beartype)
def is_psd(A: Num[Array, "n n"]) -> DTypeLike:
    """Check if a matrix is positive semi-definite.

    Parameters
    ----------
    A : Num[Array, "n n"]
        The matrix to check.

    Returns
    -------
    bool
        True if the matrix is positive semidefinite, False otherwise.
    """
    return jnp.all(jnp.linalg.eigvals(A) >= 0)


@partial(jit, static_argnames=("itmax",))
@jaxtyped(typechecker=beartype)
def schur(
    T: Num[Array, "n n"], itmax: int = 100
) -> tuple[Num[Array, "n n"], Num[Array, "n n"]]:
    """Compute the schur decomposition of a matrix using the shifted QR algorithm.

    Parameters
    ----------
    T : Num[Array, "n n"]
        The matrix to decompose.
    itmax : int
        The maximum number of iterations to perform.

    Returns
    -------
    tuple[Num[Array, "n n"], Num[Array, "n n"]]
        The upper triangular matrix and the orthogonal matrix.

    Notes
    -----
    An alternative to this function is the `scipy.linalg.schur` function, e.g.,

    ```python
    >>> def schur_scipy(T):
    ...     dtype = jax.numpy.result_type(T)
    ...     return jax.pure_callback(scipy.linalg.schur, jax.ShapeDtypeStruct(T.shape, dtype), T)
    ```

    You can also use the `jax.scipy.linalg.schur` function, but it does not support
    GPU backends.
    """
    n = T.shape[0]
    I = jnp.eye(n)  # noqa: E741

    # QR algorithm with shifts
    def make_step(i, args):
        A, Q_t = args

        shift = A[-1, -1] * I
        Q, R = jnp.linalg.qr(A - shift)
        A = R @ Q + shift
        Q_t = Q_t @ Q

        return A, Q_t

    return jax.lax.fori_loop(0, itmax, make_step, (T, I))


@jit
@jaxtyped(typechecker=beartype)
def logm_so3(R: Num[Array, "3 3"]) -> Num[Array, "3 3"]:
    """Compute the matrix logarithm of a rotation matrix in SO(3).

    Parameters
    ----------
    R : Num[Array, "3 3"]
        The rotation matrix.

    Returns
    -------
    Num[Array, "3 3"]
        The matrix logarithm of `R`.

    Warnings
    --------
    This is not defined for rotations (around any axis) equal to `jnp.pi` or `-jnp.pi`.
    You can check this using, e.g.,

    ```python
    >>> jaxgm.rotation.rotation_angle(R)
    3.1415927
    ```

    In this case, you should apply a small right perturbation to the rotation matrix.
    """
    theta = rotation_angle(R)
    w_hat = jnp.zeros((3, 3))
    return jax.lax.cond(
        jnp.sin(theta) < 1e-6,
        lambda: w_hat,
        lambda: 1 / 2 * theta / jnp.sin(theta) * (R - R.T),
    )


@jit
@jaxtyped(typechecker=beartype)
def _left_jac_inv_so3(w: Num[Array, "3"]) -> Num[Array, "3 3"]:
    """Compute the inverse of the left SO(3) Jacobian.

    Parameters
    ----------
    w : Num[Array, "3"]
        The angular velocity / exponential coordinates of the rotation.

    Returns
    -------
    Num[Array, "3 3"]
        The closed-form solution to the inverse of the left Jacobian.
    """
    jac = jnp.eye(3)
    return jax.lax.cond(
        jnp.isclose(damped_norm(w), 0.0),
        lambda: jac,
        lambda: jac
        - 0.5 * skew3(w)
        + (
            (1 / (damped_norm(w) ** 2))
            - (1 + jnp.cos(damped_norm(w)))
            / (2 * damped_norm(w) * jnp.sin(damped_norm(w)))
        )
        * (skew3(w) @ skew3(w)),
    )


@jit
@jaxtyped(typechecker=beartype)
def logm_se3(T: Num[Array, "4 4"]) -> Num[Array, "4 4"]:
    """Compute the matrix logarithm of an SE(3) element.

    Parameters
    ----------
    T : Num[Array, "4 4"]
        The SE(3) element.

    Returns
    -------
    Num[Array, "4 4"]
        The matrix logarithm of `T`, i.e., its exponential coordinates.

    Warnings
    --------
    This is not defined for rotations (around any axis) equal to `jnp.pi` or `-jnp.pi`.
    You can check this using, e.g.,

    ```python
    >>> jaxgm.rotation.rotation_angle(R)
    3.1415927
    ```

    In this case, you should apply a small right perturbation to the rotation matrix.
    """
    t, R = to_parameters(T)
    w_hat = logm_so3(R)
    w = vex3(w_hat)

    ξ = jnp.zeros((4, 4))
    ξ = ξ.at[:3, :3].set(w_hat)
    ξ = ξ.at[:3, 3].set(_left_jac_inv_so3(w) @ t)

    return ξ
