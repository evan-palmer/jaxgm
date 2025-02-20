from typing import Any, Callable

import jax
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
def directional_derivative(
    f: Callable[[Num[Array, "n n"], float], Num[Array, "n n"]], g: Num[Array, "n n"]
) -> Callable[..., Any]:
    return jax.jacfwd(lambda delta: f(g, delta))
