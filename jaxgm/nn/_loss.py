import jax.numpy as jnp
from beartype import beartype
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, Float, jaxtyped

from jaxgm._lie_group import GroupElement
from jaxgm.linalg._norm import damped_norm


@jaxtyped(typechecker=beartype)
def chordal_distance(g: Float[Array, "n n"], h: Float[Array, "n n"]) -> float:
    A = g @ jnp.linalg.inv(h) - jnp.eye(g.shape[-1])
    return jnp.trace(A.T @ A)


@jaxtyped(typechecker=beartype)
def rotation_angle(R1: Rotation, R2: Rotation) -> Float[Array, ""]:
    R1_mat, R2_mat = R1.as_matrix(), R2.as_matrix()
    g = R1_mat @ R2_mat.T

    # The angle of a rotation is computed as:
    # tr(R) = 1 + 2 * cos(theta)
    # |theta| = arccos((tr(R) - 1) / 2)

    # Compute the cosine of the angle using the trace
    cos = (jnp.trace(g) - 1) / 2
    cos = jnp.clip(cos, -1, 1)
    theta = jnp.arccos(cos)

    return theta


@jaxtyped(typechecker=beartype)
def orthogonal_error(R: Float[Array, "n n"]) -> Float[Array, ""]:
    return damped_norm(R @ R.T - jnp.eye(R.shape[-1]))


@jaxtyped(typechecker=beartype)
def special_orthogonal_error(R: Float[Array, "n n"]) -> Float[Array, ""]:
    return jnp.abs(jnp.linalg.det(R)) - 1


@jaxtyped(typechecker=beartype)
def geodesic_distance(g: GroupElement, h: GroupElement) -> float:
    # TODO: We need the matrix logarithm for this
    ...
