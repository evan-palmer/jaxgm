from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
@partial(jax.jit, static_argnames=("eps"))
def damped_norm(x: Num[Array, "..."], eps: float = 1e-6) -> Num[Array, "..."]:
    return jnp.sqrt(jnp.sum(x**2) + eps)
