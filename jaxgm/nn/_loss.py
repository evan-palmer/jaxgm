import jax.numpy as jnp
from beartype import beartype
from jax import jit
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Num, jaxtyped

from jaxgm.lie_algebra import vee
from jaxgm.linalg import frobenius_norm, logm_se3
from jaxgm.rotation import rotation_angle


@jit
@jaxtyped(typechecker=beartype)
def chordal_error(g: Num[Array, "n n"], h: Num[Array, "n n"]) -> DTypeLike:
    """Compute the squared chordal distance between two Lie group elements.

    Parameters
    ----------
    g : Num[Array, "n n"]
        The left element.
    h : Num[Array, &quot;n n&quot;]
        The right element.

    Returns
    -------
    DTypeLike
        The squared chordal distance between the two elements.

    References
    ----------
    .. [1] A. R. Geist, J. Frey, M. Zhobro, A. Levina, and G. Martius, "Learning with 3D
       rotations, a hitchhiker's guide to SO(3)", in *International Conference on Machine
       Learning*, 2024.
       Conference on Machine Learning*, 2024.
    .. [2] O. Alvarez-Tunon, Y. Brodskiy and E. Kayacan, "Loss it right: Euclidean and
       Riemannian Metrics in Learning-based Visual Odometry," in *International Symposium
       on Robotics*, Stuttgart, Germany, 2023, pp. 107-111.
    """
    return frobenius_norm(g @ jnp.linalg.inv(h) - jnp.eye(g.shape[-1]))


@jit
@jaxtyped(typechecker=beartype)
def rotation_error(R1: Num[Array, "n n"], R2: Num[Array, "n n"]) -> DTypeLike:
    """Compute the angle of rotation between two SO(3) matrices.

    Parameters
    ----------
    R1 : Num[Array, "n n"]
        The left rotation matrix.
    R2 : Num[Array, "n n"]
        The right rotation matrix.

    Returns
    -------
    DTypeLike
        The angle of rotation between the two matrices.

    See Also
    --------
    rotation_angle : Computes the rotation of a single SO(3) element.

    Notes
    -----
    This is sometimes referred to as the geodesic distance between two rotations.
    """
    return rotation_angle(R1 @ R2.T)


@jit
@jaxtyped(typechecker=beartype)
def orthogonal_error(R: Float[Array, "n n"]) -> DTypeLike:
    """Compute the "distance" of a matrix from the orthogonal group.

    Parameters
    ----------
    R : Float[Array, "n n"]
        The input matrix.

    Returns
    -------
    DTypeLike
        The error of the matrix from the orthogonal group.

    See Also
    --------
    special_orthogonal_error : Enforces the determinant of the matrix to be 1.

    Notes
    -----
    This can be used as a cost function to enforce orthogonality of a matrix. Note that
    this is not a true distance metric, but a measure of the error of the matrix from the
    orthogonal group. This also does not enforce "special" orthogonality, i.e., the
    determinant of the matrix is not necessarily 1.
    """
    return frobenius_norm(R @ R.T - jnp.eye(R.shape[-1]))


@jit
@jaxtyped(typechecker=beartype)
def special_orthogonal_error(R: Float[Array, "n n"]) -> DTypeLike:
    """Compute the "distance" of a matrix from the special orthogonal group.

    Parameters
    ----------
    R : Float[Array, "n n"]
        The input matrix.

    Returns
    -------
    DTypeLike
        The error of the matrix from the special orthogonal group.

    See Also
    --------
    orthogonal_error : Enforces matrix orthogonality.

    Notes
    -----
    This should be used in conjunction with the `orthogonal_error` function to enforce
    both orthogonality and special orthogonality of a matrix. For neural ODEs, this can
    provide a nice constraint to improve sample efficiency and ensure correctness of the
    learned dynamics.
    """
    return jnp.abs(jnp.linalg.det(R)) - 1


@jit
@jaxtyped(typechecker=beartype)
def geodesic_error(g: Num[Array, "n n"], h: Num[Array, "n n"]) -> DTypeLike:
    """Compute the geodesic distance between two Lie group elements.

    Parameters
    ----------
    g : Num[Array, "n n"]
        The left element.
    h : Num[Array, "n n"]
        The right element.

    Returns
    -------
    DTypeLike
        The geodesic distance between the two elements.
    """
    return vee(logm_se3(jnp.linalg.inv(g) @ h))
