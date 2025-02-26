from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jax import jit
from jax.typing import DTypeLike
from jaxtyping import Array, Num, jaxtyped


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def dampednorm(
    x: Num[Array, "n"], W: Optional[Num[Array, "n n"]] = None, eps: float = 1e-8
) -> DTypeLike:
    """Compute the damped 2-norm of a vector.

    Parameters
    ----------
    x : Num[Array, "n"]
        The input vector.
    W : Optional[Num[Array, "n n"]]
        The weight matrix. Defaults to None.
    eps : float, optional
        The damping factor, by default 1e-8.

    Returns
    -------
    DTypeLike
        The damped L2 norm of the input vector.

    Notes
    -----
    This function is implemented as a differentiable version of `jax.numpy.linalg.norm`.

    See Also
    --------
    softnorm : Also a differentiable version of `jax.numpy.linalg.norm`.

    """
    if W is None:
        return jnp.sqrt(jnp.sum(x**2) + eps)
    return jnp.sqrt(x.T @ W @ x + eps)


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def sqnorm(x: Num[Array, "n"], W: Optional[Num[Array, "n n"]] = None) -> DTypeLike:
    """Compute the squared norm of a vector.

    Parameters
    ----------
    x : Num[Array, "n"]
        The input vector.
    W : Optional[Num[Array, "n n"]]
        The weight matrix, by default None.

    Returns
    -------
    DTypeLike
        The squared norm of the input vector.

    See Also
    --------
    sqfrobnorm : The squared Frobenius norm of a matrix.
    """
    if W is None:
        return jnp.sum(x**2)
    return x.T @ W @ x


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def frobnorm(X: Num[Array, "n n"], eps: float = 1e-8) -> DTypeLike:
    """Compute the Frobenius norm of a matrix.

    Parameters
    ----------
    X : Num[Array, "n n"]
        The input matrix.
    eps : float, optional
        The damping factor, by default 1e-8.

    Returns
    -------
    DTypeLike
        The Frobenius norm of the input matrix.

    See Also
    --------
    sqfrobnorm : The squared Frobenius norm of a matrix.
    """
    return jnp.sqrt(jnp.trace(X.T @ X) + eps)


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def sqfrobnorm(
    X: Num[Array, "n n"], W: Optional[Num[Array, "n n"]] = None
) -> DTypeLike:
    """Compute the squared Frobenius norm of a matrix.

    Parameters
    ----------
    X : Num[Array, "n n"]
        The input matrix.
    W : Optional[Num[Array, "n n"]]
        The weight matrix, by default None.

    Returns
    -------
    DTypeLike
        The squared Frobenius norm of the input matrix.

    See Also
    --------
    dampednorm : A differentiable version of `jax.numpy.linalg.norm`.
    """
    if W is None:
        return jnp.trace(X.T @ X)
    return jnp.trace(X.T @ W @ X)


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def softnorm(x: Num[Array, "..."], eps: float = 1e-5) -> DTypeLike:
    """Compute the 2-norm, but if the norm is less than `eps`, return the squared norm.

    Parameters
    ----------
    x : Num[Array, "..."]
        The input vector.
    eps : float, optional
        The norm threshold, by default 1e-5.

    Returns
    -------
    DTypeLike
        The 2-norm of the input vector, or the squared norm if the norm is less than
        `eps`.

    Notes
    -----
    This function is continuous and has a derivative that is defined everywhere, but its
    derivative is discontinuous. Similar to the `damped_norm`, this is implemented as a
    differentiable version of `jax.numpy.linalg.norm`.

    See Also
    --------
    dampednorm : A differentiable version of `jax.numpy.linalg.norm`.

    References
    ----------
    [1] This is a copy and paste of code that Charles Dawson wrote for a previous project
        here: https://github.com/MIT-REALM/architect_corl_23/blob/3f497985a3c4f5f63e689ff4633bbbab3a6af49f/architect/systems/hide_and_seek/hide_and_seek.py#L15
    """
    scaled_square = lambda x: (eps * (x / eps) ** 2).sum()  # noqa: E731
    return jax.lax.cond(jnp.linalg.norm(x) >= eps, jnp.linalg.norm, scaled_square, x)


@jit
@jaxtyped(typechecker=beartype)
def se3norm(X: Num[Array, "n n"]) -> DTypeLike:
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

    return jnp.trace(X.T @ W @ X)
