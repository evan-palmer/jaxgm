from typing import Any, Callable

import jax
from beartype import beartype
from jaxtyping import jaxtyped

from jaxgm._lie_group import GroupElement


@jaxtyped(typechecker=beartype)
def directional_derivative(
    f: Callable[[GroupElement, float], GroupElement], g: GroupElement
) -> Callable[..., Any]:
    return jax.jacfwd(lambda delta: f(g, delta))
