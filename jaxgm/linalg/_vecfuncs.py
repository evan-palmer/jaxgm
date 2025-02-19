import jax.numpy as jnp
from beartype import beartype
from jax.typing import DTypeLike
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
def _ss3d(x1: DTypeLike, x2: DTypeLike, x3: DTypeLike) -> Num[Array, "3 3"]:
    return jnp.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])


@jaxtyped(typechecker=beartype)
def to_skew_symmetric(x: Num[Array, "3"]) -> Num[Array, "3 3"]:
    return _ss3d(x[0], x[1], x[2])
