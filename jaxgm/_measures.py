from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped

from jaxgm._lie_group import LieGroupElement


@jaxtyped(typechecker=beartype)
@partial(jax.jit, static_argnames=("eps"))
def _softnorm(x: Num[Array, "..."], eps: float = 1e-6) -> Num[Array, "..."]:
    return jnp.sqrt(jnp.sum(x**2) + eps)


@jaxtyped(typechecker=beartype)
def chordal_distance(g: LieGroupElement, h: LieGroupElement) -> float:
    return _softnorm(g.T @ h - jnp.eye(4))
