import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
def to_matrix(rot: Num[Array, "n n"], t: Num[Array, "n"]) -> Num[Array, "n+1 n+1"]:
    n = t.shape[0] + 1
    last_row = jnp.zeros((1, n))
    last_row = last_row.at[0, -1] = 1
    return jnp.block([[rot, t.reshape(-1, 1)], last_row])


@jaxtyped(typechecker=beartype)
def to_matrix_from_rotation(rot: Num[Array, "n n"]) -> Num[Array, "n+1 n+1"]:
    n = rot.shape[0] + 1
    last_row = jnp.zeros((1, n))
    last_row = last_row.at[0, -1] = 1
    return jnp.block([[rot.as_matrix(), jnp.zeros((3, 1))], last_row])


@jaxtyped(typechecker=beartype)
def to_matrix_from_translation(t: Num[Array, "n"]) -> Num[Array, "n+1 n+1"]:
    n = t.shape[0] + 1
    last_row = jnp.zeros((1, n))
    last_row = last_row.at[0, -1] = 1
    return jnp.block([[jnp.eye(t.shape[-1]), t.reshape(-1, 1)], last_row])


@jaxtyped(typechecker=beartype)
def to_parameters(
    g: Num[Array, "n n"],
) -> tuple[Num[Array, "n-1 n-1"], Num[Array, "n-1"]]:
    n = g.shape[0] - 1
    return g[:n, :n], g[:n, n]


@jaxtyped(typechecker=beartype)
def flatten(g: Num[Array, "n n"]) -> Num[Array, "n(n+1)"]:
    n = g.shape[0] - 1
    return jnp.concatenate([g[:n, n], g[:n, :n].flatten()])


@jaxtyped(typechecker=beartype)
def unflatten(g: Num[Array, "n(n+1)"]) -> Num[Array, "n n"]:
    n = (-1 + jnp.sqrt(1 + 4 * g.shape[0])) / 2
    t, rot = jnp.split(g, (n,))
    rot = rot.reshape(n, n)
    return to_matrix(rot, t)


@jaxtyped(typechecker=beartype)
def AD(g: Num[Array, "n n"], h: Num[Array, "n n"]) -> Num[Array, "n n"]:
    return g @ h @ jnp.linalg.inv(g)


@jaxtyped(typechecker=beartype)
def AD_inv(g: Num[Array, "n n"], h: Num[Array, "n n"]) -> Num[Array, "n n"]:
    return jnp.linalg.inv(g) @ h @ g
