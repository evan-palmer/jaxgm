import jax
import jax.numpy as jnp

import jaxgm

jnp.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # Generate 10 random samples with a left Gaussian distribution
    (l_gs, l_gcircs) = jaxgm.random.left_gaussian(key, jnp.eye(4), jnp.eye(6), 10)

    # Generate 10 random samples with a right Gaussian distribution
    (r_gs, r_gcircs) = jaxgm.random.right_gaussian(key, jnp.eye(4), jnp.eye(6), 10)

    # Compute the mean of the samples
    left_mean = jaxgm.random.mean(l_gs)
    right_mean = jaxgm.random.mean(r_gs)

    # Compute the covariance of the samples
    left_cov = jaxgm.random.covariance(l_gs, left_mean)
    right_cov = jaxgm.random.covariance(r_gs, right_mean)
