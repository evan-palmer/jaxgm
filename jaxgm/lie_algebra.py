import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped

import jaxgm


@jaxtyped(typechecker=beartype)
def to_matrix(ξ: Num[Array, "6"]) -> Num[Array, "4 4"]:
    v, w = jnp.split(ξ, 2)
    return jnp.block(
        [
            [jaxgm.linalg.to_skew_symmetric(w), v.reshape(-1, 1)],
            [jnp.zeros((1, 4))],
        ]
    )


@jaxtyped(typechecker=beartype)
def to_parameters(ξ: Num[Array, "4 4"]) -> Num[Array, "6"]:
    return jnp.array([ξ[0, 3], ξ[1, 3], ξ[2, 3], ξ[2, 1], ξ[0, 2], ξ[1, 0]])


@jaxtyped(typechecker=beartype)
def Ad(g: Num[Array, "n n"], h_circ: Num[Array, "n n"]) -> Num[Array, "n n"]:
    return g @ h_circ @ jnp.linalg.inv(g)


@jaxtyped(typechecker=beartype)
def Ad_inv(g: Num[Array, "n n"], h_circ: Num[Array, "n n"]) -> Num[Array, "n n"]:
    return jnp.linalg.inv(g) @ h_circ @ g


@jaxtyped(typechecker=beartype)
def lie_bracket(X: Num[Array, "n n"], Y: Num[Array, "n n"]) -> Num[Array, "n n"]:
    return X @ Y - Y @ X


@jaxtyped(typechecker=beartype)
def BCH(X: Num[Array, "n n"], Y: Num[Array, "n n"]) -> Num[Array, "n n"]:
    o1 = X + Y
    o2 = 0.5 * lie_bracket(X, Y)
    o3 = (1 / 12) * (
        lie_bracket(X, lie_bracket(X, Y)) + lie_bracket(Y, lie_bracket(Y, X))
    )
    o4 = (1 / 48) * (
        lie_bracket(Y, lie_bracket(X, lie_bracket(Y, X)))
        + lie_bracket(X, lie_bracket(Y, lie_bracket(Y, X)))
    )
    return o1 + o2 + o3 + o4
