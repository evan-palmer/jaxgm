from functools import partial

import jax
import jax.numpy as jnp
import scipy
from beartype import beartype
from jax import jit
from jaxtyping import Array, Complex, Num, jaxtyped

from jaxgm._lie_group import to_parameters
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
        X, Y = args
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


# @jaxtyped(typechecker=beartype)
def _is_rot_psd(T: Num[Array, "n n"]) -> bool:
    """Check if the rotation component in an SE(n) element is positive semidefinite.

    Parameters
    ----------
    T : Num[Array, "n n"]
        The SE(n) matrix whose rotation component should be checked.

    Returns
    -------
    bool
        True if the rotation matrix is positive semidefinite, False otherwise.
    """
    rot, _ = to_parameters(T)
    return is_pd(rot)


@jit
@jaxtyped(typechecker=beartype)
def logm(X: Num[Array, "n n"]) -> Complex[Array, "n n"]:
    """Matrix logarithm.

    This function implements a `pure_callback` around `scipy.linalg.logm`. We force
    the output to be `complex128` so that this function can be jitable.

    Parameters
    ----------
    X : Num[Array, "n n"]
        The matrix to compute the logarithm of.

    Returns
    -------
    Complex[Array, "n n"]
        The matrix logarithm of `X`.
    """
    # Only the eigenvalues of the rotation matrix will indicate whether the logm will
    # be real or complex. I believe that it has something to do with the Jordan block
    # of the matrix.
    return jax.lax.cond(
        _is_rot_psd(X),
        lambda: jax.pure_callback(
            scipy.linalg.logm, jax.ShapeDtypeStruct(X.shape, jnp.float64), X
        ).astype(jnp.complex128),
        lambda: jax.pure_callback(
            scipy.linalg.logm, jax.ShapeDtypeStruct(X.shape, jnp.complex128), X
        ),
    )
