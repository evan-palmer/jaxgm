import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped

from jaxgm._lie_group import GroupElement

TangentVector = Num[Array, "4 4"]
Twist = Num[Array, "6"]


@jaxtyped(typechecker=beartype)
def _skew_symmetric(x1: float, x2: float, x3: float) -> Num[Array, "3 3"]:
    return jnp.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])


@jaxtyped(typechecker=beartype)
def skew_symmetric(x: Num[Array, "3"]) -> Num[Array, "3 3"]:
    return _skew_symmetric(*x)


@jaxtyped(typechecker=beartype)
def to_matrix(ξ: Twist) -> TangentVector:
    v, w = jnp.split(ξ, 2)
    return jnp.block([[skew_symmetric(v), w.reshape(3, 1)], [0, 0, 0, 0]])


@jaxtyped(typechecker=beartype)
def to_parameters(ξ: TangentVector) -> Twist:
    return jnp.array([ξ[0, 3], ξ[1, 3], ξ[2, 3], ξ[2, 1], ξ[0, 2], ξ[1, 0]])


@jaxtyped(typechecker=beartype)
def Ad(g: GroupElement, h_circ: TangentVector) -> TangentVector:
    return g @ h_circ @ jnp.linalg.inv(g)


@jaxtyped(typechecker=beartype)
def Ad_inv(g: GroupElement, h_circ: TangentVector) -> TangentVector:
    return jnp.linalg.inv(g) @ h_circ @ g


@jaxtyped(typechecker=beartype)
def lie_bracket(X: TangentVector, Y: TangentVector) -> TangentVector:
    return X @ Y - Y @ X


@jaxtyped(typechecker=beartype)
def BCH(X: TangentVector, Y: TangentVector) -> TangentVector:
    return (
        X
        + Y
        + 0.5 * lie_bracket(X, Y)
        + 1 / 12 * lie_bracket(X, lie_bracket(X, Y))
        + 1 / 12 * lie_bracket(Y, lie_bracket(X, Y))
    )
