from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import jit
from jaxtyping import Array, Num, jaxtyped

from jaxgm.linalg._norm import squared_frobenius_norm


@partial(jit, static_argnames=("tol",))
@jaxtyped(typechecker=beartype)
def sqrtm_pd(A: Num[Array, "n n"], tol: float = 1e-6) -> Num[Array, "n n"]:
    """Square root of a positive definite matrix.

    This function computes the square root of a positive definite matrix using
    Denman-Beaver's scaled Newton iteration.

    Parameters
    ----------
    A : Num[Array, "n n"]
        The matrix to compute the square root of.
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
        return squared_frobenius_norm(jnp.linalg.matrix_power(X, 2) - A) > tol

    X, Y = jax.lax.while_loop(cond, make_step, (X, Y))

    return X


# @jaxtyped(typechecker=beartype)
def is_pd(A: Num[Array, "n n"]) -> bool:
    """Check if a matrix is positive definite.

    This function checks if a matrix is positive definite by computing its
    eigenvalues and checking if they are all positive.

    Parameters
    ----------
    A : Num[Array, "n n"]
        The matrix to check.

    Returns
    -------
    bool
        True if the matrix is positive definite, False otherwise.
    """
    return jnp.all(jnp.linalg.eigvalsh(A) > 0)


# @jaxtyped(typechecker=beartype)
def is_psd(A: Num[Array, "n n"]) -> bool:
    """Check if a matrix is positive semidefinite.

    This function checks if a matrix is positive semidefinite by computing its
    eigenvalues and checking if they are all non-negative.

    Parameters
    ----------
    A : Num[Array, "n n"]
        The matrix to check.

    Returns
    -------
    bool
        True if the matrix is positive semidefinite, False otherwise.
    """
    return jnp.all(jnp.linalg.eigvalsh(A) >= 0)


@partial(jit, static_argnames=("itmax",))
@jaxtyped(typechecker=beartype)
def schur(
    T: Num[Array, "n n"], itmax: int = 100
) -> tuple[Num[Array, "n n"], Num[Array, "n n"]]:
    n = T.shape[0]
    I = jnp.eye(n)  # noqa: E741

    def make_step(i, args):
        A, Q_t = args

        shift = A[-1, -1] * I
        Q, R = jnp.linalg.qr(A - shift)
        A = R @ Q + shift
        Q_t = Q_t @ Q

        return A, Q_t

    return jax.lax.fori_loop(0, itmax, make_step, (T, I))
