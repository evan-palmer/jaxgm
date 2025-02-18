from typing import Union

import jax.numpy as jnp
from beartype import beartype
from jax.typing import DTypeLike
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
def _ss3d(x1: DTypeLike, x2: DTypeLike, x3: DTypeLike) -> Num[Array, "3 3"]:
    return jnp.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])


@jaxtyped(typechecker=beartype)
def _ss2d(x: DTypeLike) -> Num[Array, "2 2"]:
    return jnp.array([[0, -x], [x, 0]])


@jaxtyped(typechecker=beartype)
def to_skew_symmetric(
    x: Union[Num[Array, "3"], Union[Num[Array, "1"]], DTypeLike],
) -> Union[Num[Array, "3 3"], Num[Array, "2 2"]]:
    if x.size == 3:
        return _ss3d(x[0], x[1], x[2])
    elif x.size == 1:
        return _ss2d(x)

    raise ValueError("Vector must be of size 1 or 3.")
