from typing import Any, Callable

import jax
from beartype import beartype
from jaxtyping import Array, Num, jaxtyped


@jaxtyped(typechecker=beartype)
def directional_derivative(
    f: Callable[[Num[Array, "n n"], float], Num[Array, "n n"]], g: Num[Array, "n n"]
) -> Callable[..., Any]:
    """Compute the directional derivative of a function.

    Parameters
    ----------
    f : Callable[[Num[Array, "n n"], float], Num[Array, "n n"]]
        The function to compute the directional derivative of.
    g : Num[Array, &quot;n n&quot;]
        The element to compute the derivative at.

    Returns
    -------
    Callable[..., Any]
        The directional derivative of the function.

    Examples
    --------
    >>> def f(q: Num[Array, "n n"], rho: float) -> Num[Array, "n n"]:
    ...     return (1 + rho / jnp.linalg.norm(q)) * q
    >>> g = jnp.eye(3)
    >>> df = directional_derivative(f, g)
    >>> df(0.0)
    """
    return jax.jacfwd(lambda delta: f(g, delta))
