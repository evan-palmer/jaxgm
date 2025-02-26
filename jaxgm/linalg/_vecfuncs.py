import jax.numpy as jnp
from beartype import beartype
from jax.typing import DTypeLike
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
def _ss2(x: DTypeLike) -> Num[Array, "2 2"]:
    # Differentiable version of `skew2`
    return jnp.array([[0, -x], [x, 0]])


@jaxtyped(typechecker=beartype)
def _ss3(x1: DTypeLike, x2: DTypeLike, x3: DTypeLike) -> Num[Array, "3 3"]:
    # Differentiable version of `skew3`
    return jnp.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])


@jaxtyped(typechecker=beartype)
def skew2(x: DTypeLike) -> Num[Array, "2 2"]:
    """Create a skew-symmetric matrix from a number.

    Parameters
    ----------
    x : DTypeLike
        The number to create the matrix from.

    Returns
    -------
    Num[Array, "2 2"]
        The skew-symmetric matrix.
    """
    return _ss2(x[2])


@jaxtyped(typechecker=beartype)
def skew3(x: Num[Array, "3"]) -> Num[Array, "3 3"]:
    """Create a skew-symmetric matrix from a vector.

    Parameters
    ----------
    x : Num[Array, "3"]
        The vector to create the matrix from.

    Returns
    -------
    Num[Array, "3 3"]
        The skew-symmetric matrix.
    """
    return _ss3(x[0], x[1], x[2])


@jaxtyped(typechecker=beartype)
def vex2(X: Num[Array, "2 2"]) -> DTypeLike:
    """Retrieve the scalar form of a 2x2 skew-symmetric matrix.

    Parameters
    ----------
    X : Num[Array, "2 2"]
        The skew-symmetric matrix.

    Returns
    -------
    DTypeLike
        The scalar form of the matrix.
    """
    return X[1, 0]


@jaxtyped(typechecker=beartype)
def vex3(X: Num[Array, "3 3"]) -> Num[Array, "3"]:
    """Retrieve the vector form of a 3x3 skew-symmetric matrix.

    Parameters
    ----------
    X : Num[Array, "3 3"]
        The skew-symmetric matrix.

    Returns
    -------
    Num[Array, "3"]
        The vector form of the matrix.
    """
    return jnp.array([X[2, 1], X[0, 2], X[1, 0]])
