import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, PRNGKeyArray, jaxtyped

from jaxgm._lie_algebra import to_matrix
from jaxgm._lie_group import LieGroupElement


@jaxtyped(typechecker=beartype)
def normal(
    key: PRNGKeyArray, mean: LieGroupElement, num_samples: int
) -> tuple[Num[Array, "n 4 4"], Num[Array, "n 4 4"]]:
    vels = jax.random.multivariate_normal(key, jnp.zeros(6), jnp.eye(6), (num_samples,))
    gs = jax.vmap(lambda ξ: mean @ jax.scipy.linalg.expm(ξ))(vels)
    g_circs = jax.vmap(to_matrix)(vels)
    return gs, g_circs
