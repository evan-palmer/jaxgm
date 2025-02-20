from typing import Union

import jax.numpy as jnp
from beartype import beartype
from jax import jit
from jaxtyping import Array, Num, jaxtyped

import jaxgm


@jaxtyped(typechecker=beartype)
def hat(ξ: Num[Array, "6"]) -> Num[Array, "4 4"]:
    """Linear mapping from the vector space R^6 to the Lie algebra se(3).

    Parameters
    ----------
    ξ : Num[Array, "6"]
        The 6-vector.

    Returns
    -------
    Num[Array, "4 4"]
        The se(3) element.
    """
    v, w = jnp.split(ξ, 2)
    return jnp.block(
        [
            [jaxgm.linalg.skew3(w), v.reshape(-1, 1)],
            [jnp.zeros((1, 4))],
        ]
    )


@jaxtyped(typechecker=beartype)
def vee(ξ: Num[Array, "4 4"]) -> Num[Array, "6"]:
    """Linear mapping from the Lie algebra se(3) to the vector space R^6.

    Parameters
    ----------
    ξ : Num[Array, "4 4"]
        The se(3) element.

    Returns
    -------
    Num[Array, "6"]
        The 6-vector.
    """
    return jnp.array([ξ[0, 3], ξ[1, 3], ξ[2, 3], ξ[2, 1], ξ[0, 2], ξ[1, 0]])


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
def lie_bracket(X: Num[Array, "n n"], Y: Num[Array, "n n"]) -> Num[Array, "n n"]:
    """Compute the Lie bracket of two Lie algebra elements.

    Parameters
    ----------
    X : Num[Array, "n n"]
        The first Lie algebra element.
    Y : Num[Array, "n n"]
        The second Lie algebra element.

    Returns
    -------
    Num[Array, "n n"]
        The Lie bracket of the two elements.
    """
    return X @ Y - Y @ X


@jit
@jaxtyped(typechecker=beartype)
def BCH(X: Num[Array, "n n"], Y: Num[Array, "n n"]) -> Num[Array, "n n"]:
    """Compute the Baker-Campbell-Hausdorff formula for two Lie algebra elements.

    Parameters
    ----------
    X : Num[Array, "n n"]
        The first Lie algebra element.
    Y : Num[Array, "n n"]
        The second Lie algebra element.

    Returns
    -------
    Num[Array, "n n"]
        The result of the BCH formula applied up to the fourth order terms.
    """
    o1 = X + Y
    o2 = 0.5 * lie_bracket(X, Y)
    o3 = (1 / 12) * (
        lie_bracket(X, lie_bracket(X, Y)) + lie_bracket(Y, lie_bracket(Y, X))
    )
    o4 = (1 / 48) * (
        lie_bracket(Y, lie_bracket(X, lie_bracket(Y, X)))
        + lie_bracket(X, lie_bracket(Y, lie_bracket(Y, X)))
    )
    return o1 + o2 + o3 + o4


def split_twist(ξ: Num[Array, "6"]) -> Union[Num[Array, "3"], Num[Array, "3"]]:
    """Split a twist into its linear and angular components.

    Parameters
    ----------
    ξ : Num[Array, "6"]
        The twist.

    Returns
    -------
    Num[Array, "3"]
        The linear component.
    Num[Array, "3"]
        The angular component.
    """
    return jnp.split(ξ, 2)
