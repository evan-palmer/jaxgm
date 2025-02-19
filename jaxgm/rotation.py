import jax.numpy as jnp
from beartype import beartype
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, DTypeLike, Num, jaxtyped

from jaxgm.linalg._norm import damped_norm


@jaxtyped(typechecker=beartype)
def normalize(R: Num[Array, "n n"], eps: float = 1e-6) -> Num[Array, "n n"]:
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


@jaxtyped(typechecker=beartype)
def rotation_angle(R: Num[Array, "n n"]) -> DTypeLike:
    # The angle of a rotation is computed as:
    # tr(R) = 1 + 2 * cos(theta)
    # |theta| = arccos((tr(R) - 1) / 2)

    # Compute the cosine of the angle using the trace
    cos = (jnp.trace(R) - 1) / 2
    cos = jnp.clip(cos, -1, 1)
    theta = jnp.arccos(cos)

    return theta


# @jaxtyped(typechecker=beartype)
# def perturb(R: Num[Array, "n n"]) -> Num[Array, "n n"]:


@jaxtyped(typechecker=beartype)
def lerp(R1: Rotation, R2: Rotation, t: float) -> Rotation:
    ...
    # TODO: Implement spherical linear interpolation (SLERP)
