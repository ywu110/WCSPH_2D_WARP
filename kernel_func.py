import math
import warp as wp

dx = 0.2 # distance between points
df_frac = 1.3
H = dx * df_frac # kernel radius

CUBIC_CONST_2D = 10.0 / (7.0 * math.pi * H ** 2)


@wp.func
def cubicKernel(r: float, h: float) -> float:
    k = CUBIC_CONST_2D
    q = r / h
    assert q >= 0.0
    res = 0.0
    if q <= 1.0:
        res = k * (1.0 - 1.5 * q ** 2.0 + 0.75 * q ** 3.0)
    elif q < 2.0:
        res = k * 0.25 * (2.0 - q) ** 3.0
    return res


@wp.func
def cubicKernelDerivative(r: float, h: float) -> float:
    k = CUBIC_CONST_2D
    q = r / h
    assert q > 0.0
    res = 0.0
    if q <= 1.0:
        res = (k / h) * (-3.0 * q + 2.25 * q ** 2.0)
    elif q < 2.0:
        res = -0.75 * (k / h) * (2.0 - q) ** 2.0
    return res
