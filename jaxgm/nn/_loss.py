import jax.numpy as jnp
from beartype import beartype
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Num, jaxtyped

from jaxgm.linalg._norm import damped_norm
from jaxgm.rotation import rotation_angle


@jaxtyped(typechecker=beartype)
def chordal_distance(g: Num[Array, "n n"], h: Num[Array, "n n"]) -> DTypeLike:
    A = g @ jnp.linalg.inv(h) - jnp.eye(g.shape[-1])
    return jnp.trace(A.T @ A)


@jaxtyped(typechecker=beartype)
def rotation_error(R1: Num[Array, "n n"], R2: Num[Array, "n n"]) -> DTypeLike:
    return rotation_angle(R1 @ R2.T)


@jaxtyped(typechecker=beartype)
def orthogonal_error(R: Float[Array, "n n"]) -> Float[Array, ""]:
    return damped_norm(R @ R.T - jnp.eye(R.shape[-1]))


@jaxtyped(typechecker=beartype)
def special_orthogonal_error(R: Float[Array, "n n"]) -> Float[Array, ""]:
    return jnp.abs(jnp.linalg.det(R)) - 1


# @jaxtyped(typechecker=beartype)
# def geodesic_distance(g: SE3GroupElement, h: SE3GroupElement) -> float:
#     # TODO: We need the matrix logarithm for this
#     ...
