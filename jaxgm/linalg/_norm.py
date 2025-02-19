from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jax.typing import DTypeLike
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
@partial(jax.jit, static_argnames=("eps"))
def damped_norm(x: Num[Array, "..."], eps: float = 1e-6) -> Num[Array, "..."]:
    return jnp.sqrt(jnp.sum(x**2) + eps)


@jaxtyped(typechecker=beartype)
def squared_frobenius_norm(X: Num[Array, "..."]) -> DTypeLike:
    return jnp.linalg.trace(X.T @ X)


@jaxtyped(typechecker=beartype)
def normest(T: Num[Array, "..."], p: int) -> float:
    T = jnp.linalg.matrix_power(T - jnp.eye(T.shape[0], dtype=T.dtype), p)

    def onenormest(A):
        return jnp.linalg.norm(A, 1, axis=(-2, -1))

    return onenormest(T)
