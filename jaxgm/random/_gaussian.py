import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Num, PRNGKeyArray, ScalarLike, jaxtyped

import jaxgm
import jaxgm.lie_algebra


@jaxtyped(typechecker=beartype)
def _sample_lie_algebra(
    key: PRNGKeyArray, cov: Num[Array, "6 6"], num_samples: int
) -> Num[Array, "n 6"]:
    return jax.random.multivariate_normal(key, jnp.zeros(6), cov, (num_samples,))


@jaxtyped(typechecker=beartype)
def left_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "4 4"], cov: Num[Array, "6 6"], num_samples: int
) -> tuple[Num[Array, "samples 4 4"], Num[Array, "samples 4 4"]]:
    """Sample from the left SE(3) Gaussian distribution.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    mean : Num[Array, "4 4"]
        The mean of the distribution.
    cov : Num[Array, "6 6"]
        The covariance of the distribution. This should be defined in the Lie algebra.
    num_samples : int
        The number of samples to draw.

    Returns
    -------
    tuple[Num[Array, "samples 4 4"], Num[Array, "samples 4 4"]]
        The Lie group elements and their corresponding Lie algebra elements.

    Notes
    -----
    The mean is applied on the left side of the exponential map.
    .. math:: mu circ exp(ξ)

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
    vels = _sample_lie_algebra(key, cov, num_samples)
    g_circs = jax.vmap(jaxgm.lie_algebra.hat)(vels)
    gs = jax.vmap(lambda g_circ: mean @ jax.scipy.linalg.expm(g_circ))(g_circs)
    return gs, g_circs


@jaxtyped(typechecker=beartype)
def right_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "4 4"], cov: Num[Array, "6 6"], num_samples: int
) -> tuple[Num[Array, "samples 4 4"], Num[Array, "samples 4 4"]]:
    """Sample from the right SE(3) Gaussian distribution.

    Parameters
    ----------
    key : PRNGKeyArray
        The random key.
    mean : Num[Array, "n n"]
        The mean of the distribution.
    cov : Num[Array, "6 6"]
        The covariance of the distribution. This should be defined in the Lie algebra.
    num_samples : int
        The number of samples to draw.

    Returns
    -------
    tuple[Num[Array, "m n n"], Num[Array, "m n n"]]
        The Lie group elements and their corresponding Lie algebra elements.

    Notes
    -----
    The mean is applied on the right side of the exponential map.
    .. math:: exp(ξ) circ mu

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
    vels = _sample_lie_algebra(key, cov, num_samples)
    g_circs = jax.vmap(jaxgm.lie_algebra.hat)(vels)
    gs = jax.vmap(lambda g_circ: jax.scipy.linalg.expm(g_circ) @ mean)(g_circs)
    return gs, g_circs


@eqx.filter_jit
@jaxtyped(typechecker=beartype)
def mean(samples: Num[Array, "n 4 4"], iters: int = 100) -> Num[Array, "4 4"]:
    """Compute the mean of a set of Lie group elements.

    Parameters
    ----------
    samples : Num[Array, "samples 4 4"]
        The set of Lie group elements.
    iters : int
        The number of iterations to compute the mean.
    tol : float
        The tolerance used to determine if the mean is valid.

    Returns
    -------
    Num[Array, "4 4"]
        The mean of the set.

    See Also
    --------
    covariance : Compute the covariance of a set of Lie group elements.
    check_mean : Compute the error between a set of Lie group elements and its mean.
    """

    # Initialize the mean
    # This makes an initial guess based on the mean of the Lie algebra elements
    def init_mean(carry, _):
        mean, exp_mean = carry
        return (mean @ jaxgm.lie_algebra.from_exp_coordinates(exp_mean), exp_mean), None

    exp_coords = jax.vmap(lambda g: jaxgm.lie_algebra.to_exp_coordinates(g))(samples)
    exp_mean = jnp.mean(exp_coords, axis=0)
    (mean, _), _ = jax.lax.scan(init_mean, (jnp.eye(4), exp_mean), None, length=5)

    # Refine the mean
    def refine_mean(carry, _):
        mean, gs = carry
        exp_coords = jax.vmap(
            lambda g: jaxgm.lie_algebra.to_exp_coordinates(jnp.linalg.inv(mean) @ g)
        )(gs)
        exp_mean = jnp.mean(exp_coords, axis=0)
        return (mean @ jaxgm.lie_algebra.from_exp_coordinates(exp_mean), gs), None

    (mean, _), _ = jax.lax.scan(refine_mean, (mean, samples), None, length=iters)

    return mean


def check_mean(samples: Num[Array, "n 4 4"], mean: Num[Array, "4 4"]) -> ScalarLike:
    """Compute the error between a set of Lie group elements and its mean.

    Parameters
    ----------
    samples : Num[Array, "n 4 4"]
        The set of Lie group elements
    mean : Num[Array, "4 4"]
        The mean of the set of Lie group elements

    Returns
    -------
    ScalarLike
        The magnitude of the error between the set of elements and the mean.

    See Also
    --------
    mean : Compute the mean of a set of Lie group elements.
    covariance : Compute the covariance of a set of Lie group elements.

    Notes
    -----
    The magnitude of the error is computed as the sum of the exponential coordinates of
    the difference between the mean and each element in the set. This can be used to
    determine the convergence of the mean estimate. A good tolerance is around 1e-5.
    If you notice that the magnitude is greater than this, you may need to increase the
    number of iterations used to compute the mean or decrease the covariance of the set.
    """
    errs = jax.vmap(
        lambda g: jaxgm.lie_algebra.to_exp_coordinates(jnp.linalg.inv(mean) @ g)
    )(samples)
    return jaxgm.linalg.softnorm(jnp.sum(errs, axis=0))


def covariance(
    samples: Num[Array, "n 4 4"], mean: Num[Array, "4 4"]
) -> Num[Array, "6 6"]:
    """Compute the covariance of a set of Lie group elements.

    Parameters
    ----------
    samples : Num[Array, "m n n"]
        The set of Lie group elements.
    mean : Num[Array, "n n"]
        The mean of the set of Lie group elements.

    Returns
    -------
    Num[Array, "n n"]
        The covariance of the set of Lie group elements.

    See Also
    --------
    mean : Compute the mean of a set of Lie group elements.

    Notes
    -----
    This is the covariance of the distribution in the Lie algebra space.
    """
    mean_inv = jnp.linalg.inv(mean)
    ys = jax.vmap(lambda g: jaxgm.lie_algebra.to_exp_coordinates(mean_inv @ g))(samples)
    sigma = jnp.einsum("ni,nj->ij", ys, ys) / len(samples)
    return sigma
