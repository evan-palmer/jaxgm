import jax.numpy as jnp
from beartype import beartype
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, Num, jaxtyped

GroupElement = Num[Array, "4 4"]
FlattenedGroupElement = Num[Array, "12"]


@jaxtyped(typechecker=beartype)
def to_matrix(rot: Rotation, t: Num[Array, "3"]) -> GroupElement:
    return jnp.block([[rot.as_matrix(), t.reshape(3, 1)], [0, 0, 0, 1]])


@jaxtyped(typechecker=beartype)
def to_parameters(g: GroupElement) -> tuple[Rotation, Num[Array, "3"]]:
    rot, t = Rotation.from_matrix(g[:3, :3]), g[:3, 3]
    return rot, t


@jaxtyped(typechecker=beartype)
def flatten(g: GroupElement) -> FlattenedGroupElement:
    return jnp.concatenate([g[:3, 3], g[:3, :3].flatten()])


@jaxtyped(typechecker=beartype)
def unflatten(g: FlattenedGroupElement) -> GroupElement:
    rot, t = Rotation.from_matrix(g[3:].reshape(3, 3)), g[:3]
    return to_matrix(rot, t)


@jaxtyped(typechecker=beartype)
def AD(g: GroupElement, h: GroupElement) -> GroupElement:
    return g @ h @ jnp.linalg.inv(g)


@jaxtyped(typechecker=beartype)
def AD_inv(g: GroupElement, h: GroupElement) -> GroupElement:
    return jnp.linalg.inv(g) @ h @ g
