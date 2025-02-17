import jax.numpy as jnp
from beartype import beartype
from jax.scipy.spatial.transform import Rotation
from jaxtyping import jaxtyped

from jaxgm.linalg._norm import damped_norm


@jaxtyped(typechecker=beartype)
def normalize(R: Rotation, eps: float = 1e-6) -> Rotation:
    """Normalize a rotation matrix.

    Args:
        R: The rotation matrix to normalize.
        eps: The damping factor.

    Returns:
        The normalized rotation matrix.
    """
    R_mat = R.as_matrix()

    x_raw, y_raw, _ = jnp.split(R_mat, 3)

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

    return Rotation.from_matrix(R_norm)


@jaxtyped(typechecker=beartype)
def lerp(R1: Rotation, R2: Rotation, t: float) -> Rotation:
    ...
    # TODO: Implement spherical linear interpolation (SLERP)
