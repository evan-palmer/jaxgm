import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, PRNGKeyArray, jaxtyped

import jaxgm
import jaxgm.lie_algebra


@jaxtyped(typechecker=beartype)
def _sample_lie_algebra(key: PRNGKeyArray, num_samples: int) -> Num[Array, "n 6"]:
    return jax.random.multivariate_normal(key, jnp.zeros(6), jnp.eye(6), (num_samples,))


@jaxtyped(typechecker=beartype)
def left_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "n n"], num_samples: int
) -> tuple[Num[Array, "m n n"], Num[Array, "m n n"]]:
    vels = _sample_lie_algebra(key, num_samples)
    g_circs = jax.vmap(jaxgm.lie_algebra.to_matrix)(vels)
    gs = jax.vmap(lambda g_circ: mean @ jax.scipy.linalg.expm(g_circ))(g_circs)
    return gs, g_circs


@jaxtyped(typechecker=beartype)
def right_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "n n"], num_samples: int
) -> tuple[Num[Array, "m n n"], Num[Array, "m n n"]]:
    vels = _sample_lie_algebra(key, num_samples)
    g_circs = jax.vmap(jaxgm.lie_algebra.to_matrix)(vels)
    gs = jax.vmap(lambda g_circ: jax.scipy.linalg.expm(g_circ) @ mean)(g_circs)
    return gs, g_circs
