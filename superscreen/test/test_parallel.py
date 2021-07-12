import os
import contextlib
import tempfile

os.environ["RAY_START_REDIS_WAIT_RETRIES"] = "48"

import pytest
import ray

import superscreen as sc


@pytest.fixture(scope="module")
def ray_initialized():
    ray.init(num_cpus=2, log_to_driver=False)
    try:
        yield True
    finally:
        ray.shutdown()


@pytest.fixture(scope="module")
def device():

    layers = [
        sc.Layer("layer0", london_lambda=1, thickness=0.1, z0=0),
        sc.Layer("layer1", london_lambda=2, thickness=0.05, z0=0.5),
    ]

    films = [
        sc.Polygon("disk", layer="layer0", points=sc.geometry.circle(5)),
        sc.Polygon("ring1", layer="layer1", points=sc.geometry.circle(4)),
    ]

    holes = [
        sc.Polygon("ring1_hole", layer="layer1", points=sc.geometry.circle(2)),
    ]

    device = sc.Device("device", layers=layers, films=films, holes=holes)
    device.make_mesh(min_triangles=1500)

    return device


def test_ray_initialized(ray_initialized):
    assert ray_initialized


@pytest.mark.parametrize("return_solutions", [False, True])
@pytest.mark.parametrize("keep_only_final_solution", [False, True])
@pytest.mark.parametrize("save", [False, True])
def test_solve_many(
    device, return_solutions, keep_only_final_solution, save, ray_initialized
):

    applied_field = sc.sources.ConstantField(0)

    circulating_currents = [{"ring1_hole": f"{i} uA"} for i in range(2)]

    context = tempfile.TemporaryDirectory() if save else contextlib.nullcontext(None)
    with context as directory:
        solutions_serial, paths_serial = sc.solve_many(
            device=device,
            parallel_method=None,
            applied_fields=applied_field,
            circulating_currents=circulating_currents,
            coupled=True,
            iterations=2,
            return_solutions=return_solutions,
            keep_only_final_solution=keep_only_final_solution,
            directory=directory,
        )

        solutions, paths = solutions_serial, paths_serial
        if save:
            assert paths is not None
            dirs = os.listdir(directory)
            assert device.name in dirs
            assert len(dirs) == len(circulating_currents) + 1
        else:
            assert paths is None
        if return_solutions:
            assert isinstance(solutions, list)
            assert len(solutions) == len(circulating_currents)
            if keep_only_final_solution:
                assert all(isinstance(s, sc.BrandtSolution) for s in solutions)
            else:
                assert all(isinstance(lst, list) for lst in solutions)
                assert all(isinstance(s, sc.BrandtSolution) for s in solutions[0])
        else:
            assert solutions is None

    context = tempfile.TemporaryDirectory() if save else contextlib.nullcontext(None)
    with context as directory:
        solutions_mp, paths_mp = sc.solve_many(
            device=device,
            parallel_method="mp",
            applied_fields=applied_field,
            circulating_currents=circulating_currents,
            coupled=True,
            iterations=2,
            return_solutions=return_solutions,
            keep_only_final_solution=keep_only_final_solution,
            directory=directory,
        )

        solutions, paths = solutions_mp, paths_mp
        if save:
            assert paths is not None
            dirs = os.listdir(directory)
            assert device.name in dirs
            assert len(dirs) == len(circulating_currents) + 1
        else:
            assert paths is None
        if return_solutions:
            assert isinstance(solutions, list)
            assert len(solutions) == len(circulating_currents)
            if keep_only_final_solution:
                assert all(isinstance(s, sc.BrandtSolution) for s in solutions)
            else:
                assert all(isinstance(lst, list) for lst in solutions)
                assert all(isinstance(s, sc.BrandtSolution) for s in solutions[0])
        else:
            assert solutions is None

    context = tempfile.TemporaryDirectory() if save else contextlib.nullcontext(None)
    with context as directory:
        solutions_ray, paths_ray = sc.solve_many(
            device=device,
            parallel_method="ray",
            applied_fields=applied_field,
            circulating_currents=circulating_currents,
            coupled=True,
            iterations=2,
            return_solutions=return_solutions,
            keep_only_final_solution=keep_only_final_solution,
            directory=directory,
        )

        solutions, paths = solutions_ray, paths_ray
        if save:
            assert paths is not None
            dirs = os.listdir(directory)
            assert device.name in dirs
            assert len(dirs) == len(circulating_currents) + 1
        else:
            assert paths is None
        if return_solutions:
            assert isinstance(solutions, list)
            assert len(solutions) == len(circulating_currents)
            if keep_only_final_solution:
                assert all(isinstance(s, sc.BrandtSolution) for s in solutions)
            else:
                assert all(isinstance(lst, list) for lst in solutions)
                assert all(isinstance(s, sc.BrandtSolution) for s in solutions[0])
        else:
            assert solutions is None

        if return_solutions:
            if keep_only_final_solution:
                for sol_serial, sol_mp, sol_ray in zip(
                    solutions_serial, solutions_mp, solutions_ray
                ):
                    assert sol_serial.equals(sol_mp, require_same_timestamp=False)
                    assert sol_serial.equals(sol_ray, require_same_timestamp=False)
                    assert sol_ray.equals(sol_mp, require_same_timestamp=False)
            else:
                for solutions in zip(solutions_serial, solutions_mp, solutions_ray):
                    for sol_serial, sol_mp, sol_ray in zip(*solutions):
                        assert sol_serial.equals(sol_mp, require_same_timestamp=False)
                        assert sol_serial.equals(sol_ray, require_same_timestamp=False)
                        assert sol_ray.equals(sol_mp, require_same_timestamp=False)
