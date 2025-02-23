import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped

from jaxgm.linalg._vecfuncs import skew3


@jaxtyped(typechecker=beartype)
def to_matrix(t: Num[Array, "n"], rot: Num[Array, "n n"]) -> Num[Array, "n+1 n+1"]:
    """Convert a translation and rotation into an SE(n) matrix.

    Parameters
    ----------
    t : Num[Array, "n"]
        The translation vector.
    rot : Num[Array, "n n"]
        The SO(n) rotation matrix.

    Returns
    -------
    Num[Array, "n+1 n+1"]
        The SE(n) matrix.
    """
    n = t.shape[0] + 1
    last_row = jnp.zeros((1, n))
    last_row = last_row.at[0, -1].set(1)
    return jnp.block([[rot, t.reshape(-1, 1)], [last_row]])


@jaxtyped(typechecker=beartype)
def to_parameters(
    g: Num[Array, "n n"],
) -> tuple[Num[Array, "n-1"], Num[Array, "n-1 n-1"]]:
    """Extract the translation and rotation from an SE(n) matrix.

    Parameters
    ----------
    g : Num[Array, "n n"]
        The SE(n) matrix.

    Returns
    -------
    tuple[Num[Array, "n-1"], Num[Array, "n-1 n-1"]]
        The translation vector and the SO(n) rotation matrix.
    """
    n = g.shape[0] - 1
    return g[:n, n], g[:n, :n]


@jaxtyped(typechecker=beartype)
def flatten(g: Num[Array, "n n"]) -> Num[Array, "n(n+1)"]:  # type: ignore
    """Flatten an SE(n) matrix into a vector.

    Parameters
    ----------
    g : Num[Array, "n n"]
        The SE(n) matrix to flatten.

    Returns
    -------
    Num[Array, "n(n+1)"]
        The flattened SE(n) matrix.

    Notes
    -----
    This is most useful when working with ODE solvers that require a vector input or
    when needing to write SE(n) elements to a file.
    """
    n = g.shape[0] - 1
    return jnp.concatenate([g[:n, n], g[:n, :n].flatten()])


@jaxtyped(typechecker=beartype)
def unflatten(g: Num[Array, "n(n+1)"]) -> Num[Array, "n n"]:  # type: ignore
    """Unflatten a vector into an SE(n) matrix.

    Parameters
    ----------
    g : Num[Array, "n(n+1)"]
        The flattened SE(n) matrix.

    Returns
    -------
    Num[Array, "n n"]
        The unflattened SE(n) matrix.
    """
    n = (-1 + jnp.sqrt(1 + 4 * g.shape[0])) / 2
    t, rot = jnp.split(g, (n,))
    rot = rot.reshape(n, n)
    return to_matrix(rot, t)


@jaxtyped(typechecker=beartype)
def to_parameters_from_flattened(
    g: Num[Array, "n(n+1)"],  # type: ignore
) -> tuple[Num[Array, "n-1"], Num[Array, "n-1 n-1"]]:
    """Extract the translation and rotation from a flattened SE(n) matrix.

    Parameters
    ----------
    g : Num[Array, "n(n+1)"]
        The flattened SE(n) matrix.

    Returns
    -------
    tuple[Num[Array, "n-1"], Num[Array, "n-1 n-1"]]
        The translation vector and the SO(n) rotation matrix.
    """
    return to_parameters(unflatten(g))


@jaxtyped(typechecker=beartype)
def to_flattened_jacobian(g: Num[Array, "4 4"]) -> Num[Array, "12 6"]:
    """Convert a SE(3) matrix into a flattened "Jacobian" matrix.

    Parameters
    ----------
    g : Num[Array, "4 4"]
        The SE(3) matrix to convert into a Jacobian matrix for flattened dynamics.

    Returns
    -------
    Num[Array, "12 6"]
        The flattened Jacobian matrix.

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
