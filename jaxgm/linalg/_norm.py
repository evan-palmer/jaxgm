from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import jit
from jax.typing import DTypeLike
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
@partial(jax.jit, static_argnames=("eps"))
def damped_norm(x: Num[Array, "..."], eps: float = 1e-8) -> Num[Array, "..."]:
    """Compute the damped 2-norm of a vector.

    Parameters
    ----------
    x : Num[Array, "..."]
        The input vector.
    eps : float, optional
        The damping factor, by default 1e-8.

    Returns
    -------
    Num[Array, "..."]
        The damped L2 norm of the input vector.

    Notes
    -----
    This function is implemented as a differentiable version of `jax.numpy.linalg.norm`.

    See Also
    --------
    softnorm : Also a differentiable version of `jax.numpy.linalg.norm`.

    """
    return jnp.sqrt(jnp.sum(x**2) + eps)


@jit
@jaxtyped(typechecker=beartype)
def squared_norm(x: Num[Array, "..."]) -> DTypeLike:
    """Compute the squared 2-norm of a vector.

    Parameters
    ----------
    x : Num[Array, "..."]
        The input vector.

    Returns
    -------
    DTypeLike
        The squared 2-norm of the input vector.
    """
    return jnp.sum(x**2)


@partial(jit, static_argnames=("eps"))
@jaxtyped(typechecker=beartype)
def softnorm(x: Num[Array, "..."], eps: float = 1e-5):
    """Compute the 2-norm, but if the norm is less than `eps`, return the squared norm.

    Parameters
    ----------
    x : Num[Array, "..."]
        The input vector.
    eps : float, optional
        The norm threshold, by default 1e-5.

    Returns
    -------
    Num[Array, "..."]
        The 2-norm of the input vector, or the squared norm if the norm is less than
        `eps`.

    Notes
    -----
    This function is continuous and has a derivative that is defined everywhere, but its
    derivative is discontinuous. Similar to the `damped_norm`, this is implemented as a
    differentiable version of `jax.numpy.linalg.norm`.

    Empirically, the `damped_norm` tends to perform better than this function.

    See Also
    --------
    damped_norm : A differentiable version of `jax.numpy.linalg.norm`.

    References
    ----------
    [1] This is a copy and paste of code that Charles Dawson wrote for a previous project
        here: https://github.com/MIT-REALM/architect_corl_23/blob/3f497985a3c4f5f63e689ff4633bbbab3a6af49f/architect/systems/hide_and_seek/hide_and_seek.py#L15
    """
    scaled_square = lambda x: (eps * (x / eps) ** 2).sum()  # noqa: E731
    return jax.lax.cond(jnp.linalg.norm(x) >= eps, jnp.linalg.norm, scaled_square, x)


@jit
@jaxtyped(typechecker=beartype)
def frobenius_norm(X: Num[Array, "n n"]) -> DTypeLike:
    """Compute the squared Frobenius norm of a matrix.

    Parameters
    ----------
    X : Num[Array, "n n"]
        The input matrix.

    Returns
    -------
    DTypeLike
        The squared Frobenius norm of the input matrix.
    """
    return jnp.linalg.trace(X.T @ X)


@jit
@jaxtyped(typechecker=beartype)
def weighted_frobenius_norm(X: Num[Array, "n n"], W: Num[Array, "n n"]) -> DTypeLike:
    """Compute the squared, weighted Frobenius norm of a matrix.

    Parameters
    ----------
    X : Num[Array, "n n"]
        The input matrix.
    W : Num[Array, "n n"]
        The weight matrix. This should be a positive diagonal matrix.

    Returns
    -------
    DTypeLike
        The weighted Frobenius norm of the input matrix.

    See Also
    --------
    weighted_se3_norm : A special case of this function for SE(3) elements using the
        inertia matrix of a unit sphere of unit mass as the weight.
    """
    return jnp.linalg.norm(X.T @ W @ X)


@jit
@jaxtyped(typechecker=beartype)
def weighted_se3_norm(X: Num[Array, "n n"]) -> DTypeLike:
    """Compute the squared, weighted Frobenius norm of an SE(3) element.

    Parameters
    ----------
    X : Num[Array, "n n"]
        The input SE(3) element.

    Returns
    -------
    DTypeLike
        The norm result.

    Notes
    -----
    This uses `weighted_frobenius_norm` with the inertia matrix for a unit sphere of
    unit mass as the norm weight.

    See Also
    --------
    weighted_frobenius_norm : The general weighted, squared Frobenius norm.
    """
    # Inertia matrix for unit sphere of unit mass
    I = (2 / 5) * jnp.identity(3)  # noqa: E741
    J = (1 / 2) * jnp.trace(I) * jnp.identity(3) - I
    M = 1
    W = jnp.block([[J, jnp.zeros((3, 1))], [jnp.zeros((1, 3)), M]])

    return weighted_frobenius_norm(X, W)
