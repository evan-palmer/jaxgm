import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

import jaxgm

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=8, suppress=True, linewidth=200)


key = jax.random.PRNGKey(0)
mat = Rotation.from_euler("xyz", [90.0, 180.0, 0.0], degrees=True).as_matrix()

print(jaxgm.rotation.rotation_angle(mat))
mat = jaxgm.rotation.perturb_right(key, mat)
print(jaxgm.rotation.rotation_angle(mat))
