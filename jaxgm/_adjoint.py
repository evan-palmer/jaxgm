import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
def Ad(g: Num[Array, "n n"], h_circ: Num[Array, "n n"]) -> Num[Array, "n n"]:
    """Apply the Adjoint operation to a Lie algebra element.

    Parameters
    ----------
    g : Num[Array, "n n"]
        The Lie group element used to move the Lie algebra element.
    h_circ : Num[Array, "n n"]
        The Lie algebra element to move.

    Returns
    -------
    Num[Array, "n n"]
        The result of the Adjoint operation.

    See Also
    --------
    AD : The Adjoint operation for Lie groups.
    Ad_inv : The inverse of this operation.

    Notes
    -----
    This does not move the Lie group element associated with the tangent vector. This
    only moves the tangent vector itself.
    """
    return g @ h_circ @ jnp.linalg.inv(g)


@jaxtyped(typechecker=beartype)
def Ad_inv(g: Num[Array, "n n"], h_circ: Num[Array, "n n"]) -> Num[Array, "n n"]:
    """Apply the inverse Adjoint operation to a Lie algebra element.

    Parameters
    ----------
    g : Num[Array, "n n"]
        The Lie group element used to move the Lie algebra element.
    h_circ : Num[Array, "n n"]
        The Lie algebra element to move.

    Returns
    -------
    Num[Array, "n n"]
        The result of the inverse Adjoint operation.

    See Also
    --------
    AD_inv : The Adjoint operation for Lie groups.
    Ad : The inverse of this operation.

    Notes
    -----
    This does not move the Lie group element associated with the tangent vector. This
    only moves the tangent vector itself.
    """
    return jnp.linalg.inv(g) @ h_circ @ g


@jaxtyped(typechecker=beartype)
def AD(g: Num[Array, "n n"], h: Num[Array, "n n"]) -> Num[Array, "n n"]:
    """Perform the Adjoint operation on a Lie group element.

    Parameters
    ----------
    g : Num[Array, "n n"]
        The first Lie group element.
    h : Num[Array, "n n"]
        The second Lie group element.

    Returns
    -------
    Num[Array, "n n"]
        The result of the Adjoint operation.

    Notes
    -----
    Computes the formula:
    .. math:: g circ h circ g^{-1}
    """
    return g @ h @ jnp.linalg.inv(g)


@jaxtyped(typechecker=beartype)
def AD_inv(g: Num[Array, "n n"], h: Num[Array, "n n"]) -> Num[Array, "n n"]:
    """Perform the inverse Adjoint operation on a Lie group element

    Parameters
    ----------
    g : Num[Array, "n n"]
        The first Lie group element.
    h : Num[Array, "n n"]
        The second Lie group element.

    Returns
    -------
    Num[Array, "n n"]
        The result of the inverse Adjoint operation.

    Notes
    -----
    Computes the formula:
    .. math:: g^{-1} circ h circ g
    """
    return jnp.linalg.inv(g) @ h @ g
