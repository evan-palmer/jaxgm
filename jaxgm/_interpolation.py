from typing import Optional

import diffrax
import equinox as eqx
import jax
from jaxtyping import Array, PyTree, Real, ScalarLike, Shaped


class SphericalLinearInterpolation(diffrax.AbstractGlobalInterpolation):
    # https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_global_interpolation.p
    # https://github.com/matthew-brett/transforms3d/blob/6a43a98e3659d198ff6ce2c90d52ddef50fcf770/transforms3d/_gohlketransforms.py#L1436
    ts: Real[Array, "times"]
    ys: PyTree[Shaped[Array, "times ..."]]

    def __check_init__(self):
        def _check(_ys):
            if _ys.shape[0] != self.ts.shape[0]:
                raise ValueError(
                    "The first dimension of `ys` must match the length of `ts`."
                )

            # TODO: Verify that the ys are valid quaternions

        jax.tree_map(_check, self.ys)

    @property
    def ts_size(self) -> ScalarLike:
        return self.ts.shape[0]

    @eqx.filter_jit
    def evaluate(
        self, t0: ScalarLike, t1: Optional[ScalarLike] = None, left: bool = True
    ) -> PyTree[Array]: ...


class SVDProjectionInterpolation(diffrax.AbstractGlobalInterpolation):
    # https://github.com/ChirikjianLab/primp-python/blob/main/primp/util/interp_se3_trajectory.py
    ...
