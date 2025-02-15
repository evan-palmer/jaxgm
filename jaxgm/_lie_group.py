import jax.numpy as jnp
from beartype import beartype
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array, Num, jaxtyped

LieGroupElement = Num[Array, "4 4"]


@jaxtyped(typechecker=beartype)
def to_matrix(rot: Rotation, t: Num[Array, "3"]) -> LieGroupElement:
    return jnp.block([[rot.as_matrix(), t.reshape(3, 1)], [0, 0, 0, 1]])


@jaxtyped(typechecker=beartype)
def to_parameters(g: LieGroupElement) -> tuple[Rotation, Num[Array, "3"]]:
    rot, t = Rotation.from_matrix(g[:3, :3]), g[:3, 3]
    return rot, t


@jaxtyped(typechecker=beartype)
def flatten(g: LieGroupElement) -> Num[Array, "12"]:
    return jnp.concatenate([g[:3, 3], g[:3, :3].flatten()])


@jaxtyped(typechecker=beartype)
def unflatten(g: Num[Array, "12"]) -> LieGroupElement:
    rot, t = Rotation.from_matrix(g[3:].reshape(3, 3)), g[:3]
    return to_matrix(rot, t)


@jaxtyped(typechecker=beartype)
def AD(g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
    return g @ h @ jnp.linalg.inv(g)


@jaxtyped(typechecker=beartype)
def AD_inv(g: LieGroupElement, h: LieGroupElement) -> LieGroupElement:
    return jnp.linalg.inv(g) @ h @ g
