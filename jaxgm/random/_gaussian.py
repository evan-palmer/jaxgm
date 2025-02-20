import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, PRNGKeyArray, jaxtyped

import jaxgm
import jaxgm.lie_algebra


@jaxtyped(typechecker=beartype)
def _sample_lie_algebra(key: PRNGKeyArray, num_samples: int) -> Num[Array, "n 6"]:
    return jax.random.multivariate_normal(key, jnp.zeros(6), jnp.eye(6), (num_samples,))


@jaxtyped(typechecker=beartype)
def left_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "n n"], num_samples: int
) -> tuple[Num[Array, "m n n"], Num[Array, "m n n"]]:
    """Sample from the left SE(3) Gaussian distribution.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    mean : Num[Array, "n n"]
        The mean of the distribution.
    num_samples : int
        The number of samples to draw.

    Returns
    -------
    tuple[Num[Array, "m n n"], Num[Array, "m n n"]]
        The Lie group elements and their corresponding Lie algebra elements.

    Notes
    -----
    The mean is applied on the left side of the exponential map.
    .. math:: \mu \circ \exp(\xi)

    References
    ----------
    .. [1] A. W. Long, K. C. Wolfe, M. J. Mashner, and G. S. Chirikjian, "The Banana
       Distribution is Gaussian: A Localization Study with Exponential Coordinates", in
       *Robotics: Science and Systems*, 2012.
    .. [2] T. D. Barfoot, "State Estimation for Robotics", Cambridge University Press,
       2021.
    .. [3] G. S. Chirikjian, "Stochastic Models, Information Theory, and Lie Groups,
       Volume 2: Analytic Methods and Modern Applications", Springer, 2012.
    """
    vels = _sample_lie_algebra(key, num_samples)
    g_circs = jax.vmap(jaxgm.lie_algebra.hat)(vels)
    gs = jax.vmap(lambda g_circ: mean @ jax.scipy.linalg.expm(g_circ))(g_circs)
    return gs, g_circs


@jaxtyped(typechecker=beartype)
def right_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "n n"], num_samples: int
) -> tuple[Num[Array, "m n n"], Num[Array, "m n n"]]:
    """Sample from the right SE(3) Gaussian distribution.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    mean : Num[Array, "n n"]
        The mean of the distribution.
    num_samples : int
        The number of samples to draw.

    Returns
    -------
    tuple[Num[Array, "m n n"], Num[Array, "m n n"]]
        The Lie group elements and their corresponding Lie algebra elements.

    Notes
    -----
    The mean is applied on the right side of the exponential map.
    .. math:: \exp(\xi) \circ \mu

    References
    ----------
    .. [1] A. W. Long, K. C. Wolfe, M. J. Mashner, and G. S. Chirikjian, "The Banana
       Distribution is Gaussian: A Localization Study with Exponential Coordinates", in
       *Robotics: Science and Systems*, 2012.
    .. [2] T. D. Barfoot, "State Estimation for Robotics", Cambridge University Press,
       2021.
    .. [3] G. S. Chirikjian, "Stochastic Models, Information Theory, and Lie Groups,
       Volume 2: Analytic Methods and Modern Applications", Springer, 2012.
    """
    vels = _sample_lie_algebra(key, num_samples)
    g_circs = jax.vmap(jaxgm.lie_algebra.hat)(vels)
    gs = jax.vmap(lambda g_circ: jax.scipy.linalg.expm(g_circ) @ mean)(g_circs)
    return gs, g_circs


def approx_mean(gs: Num[Array, "m n n"]) -> Num[Array, "n n"]: ...


def approx_cov(
    gs: Num[Array, "m n n"], mean: Num[Array, "n n"]
) -> Num[Array, "n n"]: ...
