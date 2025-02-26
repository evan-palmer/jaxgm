import jax
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped

from jaxgm.lie_algebra import hat, vee
from jaxgm.linalg._matfuncs import logm_se3


@jaxtyped(typechecker=beartype)
def to_exponential_coords(g: Num[Array, "4 4"]) -> Num[Array, "6"]:
    """Convert an SE(3) matrix into exponential coordinates.

    Parameters
    ----------
    g : Num[Array, "4 4"]
        The SE(3) matrix.

    Returns
    -------
    Num[Array, "6"]
        The exponential coordinates.

    See Also
    --------
    from_exponential_coords : The inverse operation.
    """
    return vee(logm_se3(g))


@jaxtyped(typechecker=beartype)
def from_exponential_coords(ξ: Num[Array, "6"]) -> Num[Array, "4 4"]:
    """Convert exponential coordinates into an SE(3) matrix.

    Parameters
    ----------
    ξ : Num[Array, "6"]
        The exponential coordinates.

    Returns
    -------
    Num[Array, "4 4"]
        The SE(3) matrix.
    """
    return jax.scipy.linalg.expm(hat(ξ))
