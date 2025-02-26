import jax
import jax.numpy as jnp

import jaxgm

jnp.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    samples, _ = jaxgm.random.left_gaussian(key, jnp.eye(4), jnp.eye(6), 100)

    mean = jaxgm.random.mean(samples)

    error = jaxgm.random.check_mean(samples, mean)
    if error > 1e-6:
        print("Unable to compute the mean of the set. Try decreasing the covariance.")

    print(f"Mean: \n{mean}")

    norm = jaxgm.linalg.softnorm(jaxgm.to_exponential_coords(mean))
    if norm < 1.0:
        print("The distribution is concentrated.")
    else:
        print("The distribution is not concentrated.")
