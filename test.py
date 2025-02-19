import jax
import jax.numpy as jnp

import jaxgm
import jaxgm.lie_algebra
import jaxgm.lie_group

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=5, suppress=True, linewidth=200)


def _geodesic_distance(g, h, key):
    return jaxgm.lie_algebra.to_parameters(
        jaxgm.linalg.logm(jnp.linalg.inv(g) @ h, key)
    )


k1, k2 = jax.random.split(jax.random.PRNGKey(0))
ghs, _ = jaxgm.random.left_gaussian(k1, jnp.eye(4), 2)
gs, hs = jnp.split(ghs, 2, axis=0)

# print(jax.vmap(jaxgm.linalg.schur)(gs))

print(jax.vmap(_geodesic_distance)(gs, hs, jax.random.split(k2, gs.shape[0])))

# print(jax.jacfwd(_geodesic_distance, argnums=(0, 1))(g, h))
