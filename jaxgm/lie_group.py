import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped

from jaxgm.linalg._vecfuncs import skew3


@jaxtyped(typechecker=beartype)
def to_matrix(t: Num[Array, "3"], rot: Num[Array, "3 3"]) -> Num[Array, "4 4"]:
    """Convert a translation and rotation into an SE(3) matrix.

    Parameters
    ----------
    t : Num[Array, "3"]
        The translation vector.
    rot : Num[Array, "3 3"]
        The SO(3) rotation matrix.

    Returns
    -------
    Num[Array, "4 4"]
        The SE(3) matrix.
    """
    return jnp.block([[rot, t.reshape(-1, 1)], [jnp.array([0, 0, 0, 1])]])


@jaxtyped(typechecker=beartype)
def to_parameters(g: Num[Array, "4 4"]) -> tuple[Num[Array, "3"], Num[Array, "3 3"]]:
    """Extract the translation and rotation from an SE(3) matrix.

    Parameters
    ----------
    g : Num[Array, "4 4"]
        The SE(3) matrix.

    Returns
    -------
    tuple[Num[Array, "3"], Num[Array, "3 3"]]
        The translation vector and the SO(3) rotation matrix.
    """
    return g[:3, 3], g[:3, :3]


@jaxtyped(typechecker=beartype)
def flatten(g: Num[Array, "4 4"]) -> Num[Array, "12"]:
    """Flatten an SE(3) matrix into a vector.

    Parameters
    ----------
    g : Num[Array, "4 4"]
        The SE(3) matrix to flatten.

    Returns
    -------
    Num[Array, "12"]
        The flattened SE(3) matrix.

    Notes
    -----
    This is most useful when working with ODE solvers that require a vector input or
    when needing to write SE(3) elements to a file.
    """
    return jnp.concatenate([g[:3, 3], g[:3, :3].flatten()])


@jaxtyped(typechecker=beartype)
def unflatten(g: Num[Array, "12"]) -> Num[Array, "4 4"]:
    """Unflatten a vector into an SE(3) matrix.

    Parameters
    ----------
    g : Num[Array, "12"]
        The flattened SE(3) matrix.

    Returns
    -------
    Num[Array, "4 4"]
        The unflattened SE(3) matrix.
    """
    t, rot = jnp.split(g, (3,))
    rot = rot.reshape(3, 3)
    return to_matrix(t, rot)


@jaxtyped(typechecker=beartype)
def flatjac(g: Num[Array, "4 4"]) -> Num[Array, "12 6"]:
    """Convert a SE(3) matrix into a flattened "Jacobian" matrix.

    Parameters
    ----------
    g : Num[Array, "4 4"]
        The SE(3) matrix to convert into a Jacobian matrix for flattened dynamics.

    Returns
    -------
    Num[Array, "12 6"]
        The flattened Jacobian matrix.

    See Also
    --------
    flatjac2 : The equivalent function for SE(2).

    Notes
    -----
    This is useful when you are computing the Lie group dynamics and the ODE solver
    needs the output to be a flattened SE(3) element.
    """
    _, R = to_parameters(g)
    return jnp.block(
        [
            [R, jnp.zeros_like(R)],
            [jnp.zeros_like(R), skew3(R[0])],
            [jnp.zeros_like(R), skew3(R[1])],
            [jnp.zeros_like(R), skew3(R[2])],
        ]
    )
