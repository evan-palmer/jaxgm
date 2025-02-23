import jax
import jax.numpy as jnp

import jaxgm

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # Generate 2 random samples
    gs, gcircs = jaxgm.random.right_gaussian(key, jnp.eye(4), jnp.eye(6), 2)
    (gcirc1, gcirc2) = gcircs

    # Compute the Lie group dynamics
    dynamics = jax.vmap(lambda g, gcirc: g @ gcirc)(gs, gcircs)

    print(f"Lie group dynamics: {dynamics}")

    # We can also compute the flattened dynamics
    def f(g, gcirc):
        J = jaxgm.lie_group.to_flattened_jacobian(g)
        twist = jaxgm.lie_algebra.vee(gcirc)
        return J @ twist

    flattened_dynamics = jax.vmap(f)(gs, gcircs)

    print(f"Flattened dynamics: {flattened_dynamics}")
