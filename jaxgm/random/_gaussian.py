import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, PRNGKeyArray, jaxtyped

from jaxgm._lie_algebra import to_matrix


@jaxtyped(typechecker=beartype)
def _sample_lie_algebra(key: PRNGKeyArray, num_samples: int) -> Num[Array, "n 6"]:
    return jax.random.multivariate_normal(key, jnp.zeros(6), jnp.eye(6), (num_samples,))


@jaxtyped(typechecker=beartype)
def left_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "n n"], num_samples: int
) -> tuple[Num[Array, "m n n"], Num[Array, "m n n"]]:
    vels = _sample_lie_algebra(key, num_samples)
    gs = jax.vmap(lambda ξ: mean @ jax.scipy.linalg.expm(ξ))(vels)
    g_circs = jax.vmap(to_matrix)(vels)
    return gs, g_circs


@jaxtyped(typechecker=beartype)
def right_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "n n"], num_samples: int
) -> tuple[Num[Array, "m n n"], Num[Array, "m n n"]]:
    vels = _sample_lie_algebra(key, num_samples)
    gs = jax.vmap(lambda ξ: jax.scipy.linalg.expm(ξ) @ mean)(vels)
    g_circs = jax.vmap(to_matrix)(vels)
    return gs, g_circs
