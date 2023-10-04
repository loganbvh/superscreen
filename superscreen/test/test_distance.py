import numpy as np
import pytest
import scipy.spatial.distance

import superscreen as sc


@pytest.mark.parametrize("metric", ("euclidean", "sqeuclidean"))
def test_cdist_invalid_shape(metric):
    XA = np.random.random((100, 4))

    XB = np.random.random((333, 4))
    with pytest.raises(ValueError):
        sc.distance.cdist(XA, XB, metric=metric)

    XB = np.random.random((333, 2))
    with pytest.raises(ValueError):
        sc.distance.cdist(XA, XB, metric=metric)


def test_cdist_invalid_metric():
    XA = np.random.random((100, 2))
    XB = np.random.random((333, 2))

    with pytest.raises(ValueError):
        sc.distance.cdist(XA, XB, metric="invalid")


@pytest.mark.parametrize("metric", ("euclidean", "sqeuclidean"))
@pytest.mark.parametrize("dtype", ("float64", "float32"))
@pytest.mark.parametrize("ndim", (2, 3))
def test_cdist(metric, dtype, ndim):
    XA = np.random.random((100, ndim)).astype(dtype)
    XB = np.random.random((333, ndim)).astype(dtype)
    dist_sc = sc.distance.cdist(XA, XB, metric=metric)
    dist_sp = scipy.spatial.distance.cdist(XA, XB, metric=metric)
    assert np.allclose(dist_sc, dist_sp)
