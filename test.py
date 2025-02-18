import jax
import jax.numpy as jnp

import jaxgm
import jaxgm._lie_algebra

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=3, suppress=True, linewidth=200)

v = jnp.array([1, 2, 3]).astype(jnp.float64)

# jax.lax.cond(v.size == 3, lambda: jnp.eye(3), lambda: jnp.eye(4))


@jax.jit
def this_is_a_test(v):
    v = jaxgm._lie_algebra.to_matrix(v)

    g = jax.scipy.linalg.expm(v)

    jax.debug.print("{v}", v=jaxgm.linalg.is_pd(g))
    v = jaxgm.linalg.logm(g)
    return v


print(this_is_a_test(v))
