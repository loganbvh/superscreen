import numpy as np

from superscreen.device.mesh import MeshOperators


def test_C_vector_uses_bounding_box_center():
    # A rectangle whose sites are crowded on one side: the result must not
    # depend on where the sites happen to be dense.
    x, y = np.meshgrid(np.linspace(-2, 2, 21), np.linspace(-1, 1, 11))
    grid = np.column_stack([x.ravel(), y.ravel()])
    rng = np.random.default_rng(0)
    crowd = np.column_stack([rng.uniform(1, 2, 500), rng.uniform(-1, 1, 500)])
    C = MeshOperators.C_vector(np.concatenate([grid, crowd]))[: len(grid)]
    C_mirror = MeshOperators.C_vector(np.concatenate([grid * [-1, 1], -crowd]))
    C_mirror = C_mirror[: len(grid)]
    finite = C < 1e29
    np.testing.assert_allclose(C[finite], C_mirror[finite], rtol=1e-12)
