import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jax import jit
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, DTypeLike, Num, PRNGKeyArray, jaxtyped

from jaxgm.linalg._norm import damped_norm


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def normalize(R: Num[Array, "4 4"], eps: float = 1e-6) -> Num[Array, "4 4"]:
    """Normalize a rotation matrix.

    Parameters
    ----------
    R : Num[Array, "4 4"]
        The rotation matrix to normalize.
    eps : float, optional
        The epsilon value used by the damped norm. This shouldn't need to be changed, by
        default 1e-6.

    Returns
    -------
    Num[Array, "4 4"]
        The normalized rotation matrix
    """
    x_raw, y_raw, _ = jnp.split(R, 3)

    # Normalize x-axis
    x_norm = damped_norm(x_raw, eps=eps)
    x = x_raw / jnp.maximum(x_norm, 1e-8)

    # Compute Z-axis using cross product of x and y_raw
    z = jnp.cross(x, y_raw)
    z_norm = damped_norm(z, eps=eps)
    z = z / jnp.maximum(z_norm, 1e-8)

    # Compute Y-axis using cross product of z and x
    y = jnp.cross(z, x)

    # Combine axes to form the rotation matrix
    R_norm = jnp.stack((x, y, z)).squeeze()

    return R_norm


@jit
@jaxtyped(typechecker=beartype)
def rotation_angle(R: Num[Array, "4 4"]) -> DTypeLike:
    """Compute the angle of a rotation matrix.

    Parameters
    ----------
    R : Num[Array, "4 4"]
        The rotation matrix.

    Returns
    -------
    DTypeLike
        The angle of the rotation matrix

    Examples
    --------
    This can be used to check whether or not a rotation matrix is singular:

    ```python
    >>> R = Rotation.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix()
    >>> jaxgm.rotation.rotation_angle(R)
    3.1415927
    ```
    """
    # The angle of a rotation is computed as:
    # tr(R) = 1 + 2 * cos(theta)
    # |theta| = arccos((tr(R) - 1) / 2)

    # Compute the cosine of the angle using the trace
    cos = (jnp.trace(R) - 1) / 2
    cos = jnp.clip(cos, -1, 1)
    theta = jnp.arccos(cos)

    return theta


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def perturb_left(
    key: PRNGKeyArray, R: Num[Array, "4 4"], eps: float = 1e-6
) -> Num[Array, "n n"]:
    """Apply a small random perturbation to the left side of a rotation matrix.

    Parameters
    ----------
    key : PRNGKeyArray
        The PRNG key for random number generation.
    R : Num[Array, "4 4"]
        The rotation matrix to perturb.
    eps : float, optional
        The maximum perturbation angle, by default 1e-6.

    Returns
    -------
    Num[Array, "n n"]
        The perturbed rotation matrix.

    See Also
    --------
    perturb_right : Apply a small random perturbation to the right side of a rotation
        matrix.
    rotation_angle : Compute the angle of a rotation matrix.

    Notes
    -----
    This is useful when needing to apply a small perturbation to a rotation matrix (e.g.,
    when the rotation matrix is singular).
    """
    angles = jax.random.uniform(key, shape=(3,), minval=-eps, maxval=eps)
    noise = Rotation.from_euler("xyz", angles).as_matrix()
    return noise @ R


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def perturb_right(
    key: PRNGKeyArray, R: Num[Array, "4 4"], eps: float = 1e-6
) -> Num[Array, "n n"]:
    """Apply a small random perturbation to the right side of a rotation matrix.

    Parameters
    ----------
    key : PRNGKeyArray
        The PRNG key for random number generation.
    R : Num[Array, "4 4"]
        The rotation matrix to perturb.
    eps : float, optional
        The maximum perturbation angle, by default 1e-6.

    Returns
    -------
    Num[Array, "n n"]
        The perturbed rotation matrix.

    See Also
    --------
    perturb_left : Apply a small random perturbation to the left side of a rotation
        matrix.
    rotation_angle : Compute the angle of a rotation matrix.

    Notes
    -----
    This is useful when needing to apply a small perturbation to a rotation matrix (e.g.,
    when the rotation matrix is singular).
    """
    angles = jax.random.uniform(key, shape=(3,), minval=-eps, maxval=eps)
    noise = Rotation.from_euler("xyz", angles).as_matrix()
    return R @ noise
