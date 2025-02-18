import jax
import jax.numpy as jnp

import jaxgm
import jaxgm._lie_algebra
import jaxgm._lie_group

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=3, suppress=True, linewidth=200)


# jax.lax.cond(v.size == 3, lambda: jnp.eye(3), lambda: jnp.eye(4))


# @jax.jit
def this_is_a_test():
    # v = jaxgm._lie_algebra.to_matrix(v)

    g = jaxgm._lie_group.to_matrix_from_rotation(
        # Rotation.from_euler(
        #     "xyz", jnp.array([170.0, 0.0, 0.0]), degrees=True
        # ).as_matrix()
        jnp.array(
            [
                [jnp.cos(jnp.pi / 2), -jnp.sin(jnp.pi / 2)],
                [jnp.sin(jnp.pi / 2), jnp.cos(jnp.pi / 2)],
            ]
        )
    )
    # g = jax.scipy.linalg.expm(v)

    # jax.debug.print("{v}", v=g)
    # jax.debug.print("{v}", v=jaxgm.linalg.is_pd(g))
    print(g)

    print(jnp.linalg.eigvals(g))
    # print(jaxgm.linalg.is_pd(jnp.array([[0, -jnp.pi / 2], [jnp.pi / 2, 0]])))
    v = jaxgm.linalg.logm(g)
    return v


print(this_is_a_test())
