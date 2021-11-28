import os
import tempfile

import pytest
import ray

import superscreen as sc


@pytest.fixture(scope="module")
def ray_initialized():
    # Seems to help on Windows :/
    assert os.environ["RAY_START_REDIS_WAIT_RETRIES"] == "48"
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
        sc.Layer("layer2", london_lambda=3, thickness=0.05, z0=1.0),
    ]

    films = [
        sc.Polygon("disk", layer="layer0", points=sc.geometry.circle(5)),
        sc.Polygon("ring1", layer="layer1", points=sc.geometry.circle(4)),
        sc.Polygon("disk2", layer="layer2", points=sc.geometry.circle(6)),
    ]

    holes = [
        sc.Polygon("ring1_hole", layer="layer1", points=sc.geometry.circle(2)),
    ]

    device = sc.Device("device", layers=layers, films=films, holes=holes)
    device.make_mesh(min_points=1000)

    return device


def test_ray_initialized(ray_initialized):
    assert ray_initialized


@pytest.mark.parametrize(
    "vortices, return_solutions, save, use_shared_memory",
    [
        (None, False, True, False),
        (sc.Vortex(x=0, y=0, layer="layer0"), True, False, True),
        ([sc.Vortex(x=0, y=0, layer="layer0")], False, True, False),
        (
            [
                [sc.Vortex(x=0, y=0, layer="layer0")],
                [
                    sc.Vortex(x=0, y=0, layer="layer0"),
                    sc.Vortex(x=1, y=-1, layer="layer0"),
                ],
            ],
            True,
            False,
            True,
        ),
    ],
)
@pytest.mark.parametrize("keep_only_final_solution", [False, True])
def test_solve_many(
    device,
    vortices,
    return_solutions,
    keep_only_final_solution,
    save,
    use_shared_memory,
    ray_initialized,
):

    applied_field = sc.sources.ConstantField(0)

    circulating_currents = [{"ring1_hole": f"{i} uA"} for i in range(2)]

    if save:
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = sc.io.NullContextManager()
    with save_context as directory:
        solutions_serial, paths_serial = sc.solve_many(
            device=device,
            parallel_method=None,
            applied_fields=applied_field,
            circulating_currents=circulating_currents,
            vortices=vortices,
            iterations=2,
            return_solutions=return_solutions,
            keep_only_final_solution=keep_only_final_solution,
            directory=directory,
            use_shared_memory=use_shared_memory,
        )
        solver = "superscreen.solve_many:serial:1"
        solutions, paths = solutions_serial, paths_serial
        if save:
            assert paths is not None
            dirs = os.listdir(directory)
            assert len(dirs) == len(circulating_currents)
            if keep_only_final_solution:
                for d in dirs:
                    _ = sc.Solution.from_file(os.path.join(directory, d))
        else:
            assert paths is None
        if return_solutions:
            assert isinstance(solutions, list)
            assert len(solutions) == len(circulating_currents)
            if keep_only_final_solution:
                assert all(isinstance(s, sc.Solution) for s in solutions)
                assert all(s.solver == solver for s in solutions)
            else:
                assert all(isinstance(lst, list) for lst in solutions)
                assert all(isinstance(s, sc.Solution) for s in solutions[0])
                assert all(s.solver == solver for s in solutions[0])
        else:
            assert solutions is None

    if save:
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = sc.io.NullContextManager()
    with save_context as directory:
        solutions_mp, paths_mp = sc.solve_many(
            device=device,
            parallel_method="mp",
            applied_fields=applied_field,
            circulating_currents=circulating_currents,
            vortices=vortices,
            iterations=2,
            return_solutions=return_solutions,
            keep_only_final_solution=keep_only_final_solution,
            directory=directory,
            use_shared_memory=use_shared_memory,
        )
        ncpu = min(len(circulating_currents), os.cpu_count())
        solver = f"superscreen.solve_many:multiprocessing:{ncpu}"
        solutions, paths = solutions_mp, paths_mp
        if save:
            assert paths is not None
            dirs = os.listdir(directory)
            assert len(dirs) == len(circulating_currents)
            if keep_only_final_solution:
                for d in dirs:
                    _ = sc.Solution.from_file(os.path.join(directory, d))
        else:
            assert paths is None
        if return_solutions:
            assert isinstance(solutions, list)
            assert len(solutions) == len(circulating_currents)
            if keep_only_final_solution:
                assert all(isinstance(s, sc.Solution) for s in solutions)
                assert all(s.solver == solver for s in solutions)
            else:
                assert all(isinstance(lst, list) for lst in solutions)
                assert all(isinstance(s, sc.Solution) for s in solutions[0])
                assert all(s.solver == solver for s in solutions[0])
        else:
            assert solutions is None

    if save:
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = sc.io.NullContextManager()
    with save_context as directory:
        solutions_ray, paths_ray = sc.solve_many(
            device=device,
            parallel_method="ray",
            applied_fields=applied_field,
            circulating_currents=circulating_currents,
            vortices=vortices,
            iterations=2,
            return_solutions=return_solutions,
            keep_only_final_solution=keep_only_final_solution,
            directory=directory,
            use_shared_memory=use_shared_memory,
        )
        solver = "superscreen.solve_many:ray:2"
        solutions, paths = solutions_ray, paths_ray
        if save:
            assert paths is not None
            dirs = os.listdir(directory)
            assert len(dirs) == len(circulating_currents)
            if keep_only_final_solution:
                for d in dirs:
                    _ = sc.Solution.from_file(os.path.join(directory, d))
        else:
            assert paths is None
        if return_solutions:
            assert isinstance(solutions, list)
            assert len(solutions) == len(circulating_currents)
            if keep_only_final_solution:
                assert all(isinstance(s, sc.Solution) for s in solutions)
                assert all(s.solver == solver for s in solutions)
            else:
                assert all(isinstance(lst, list) for lst in solutions)
                assert all(isinstance(s, sc.Solution) for s in solutions[0])
                assert all(s.solver == solver for s in solutions[0])
        else:
            assert solutions is None

    if return_solutions:
        if keep_only_final_solution:
            for sol_serial, sol_mp, sol_ray in zip(
                solutions_serial, solutions_mp, solutions_ray
            ):
                assert sol_serial.equals(sol_mp)
                assert sol_serial.equals(sol_ray)
                assert sol_ray.equals(sol_mp)
        else:
            for solutions in zip(solutions_serial, solutions_mp, solutions_ray):
                for sol_serial, sol_mp, sol_ray in zip(*solutions):
                    assert sol_serial.equals(sol_mp)
                    assert sol_serial.equals(sol_ray)
                    assert sol_ray.equals(sol_mp)
