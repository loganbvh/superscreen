import contextlib
import os
import tempfile

import joblib
import pytest
import ray

import superscreen as sc


@pytest.fixture(scope="module")
def ray_initialized():
    # Seems to help on Windows :/
    assert os.environ["RAY_START_REDIS_WAIT_RETRIES"] == "48"
    num_cpus = min(2, joblib.cpu_count())
    ray.init(num_cpus=num_cpus, log_to_driver=False)
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
@pytest.mark.parametrize("num_cpus", [None, 2])
def test_solve_many(
    device,
    vortices,
    return_solutions,
    keep_only_final_solution,
    save,
    use_shared_memory,
    num_cpus,
    ray_initialized,
):

    applied_field = sc.sources.ConstantField(0)

    circulating_currents = [{"ring1_hole": f"{i} uA"} for i in range(2)]

    solve_kwargs = dict(
        applied_fields=applied_field,
        circulating_currents=circulating_currents,
        vortices=vortices,
        iterations=2,
        return_solutions=return_solutions,
        keep_only_final_solution=keep_only_final_solution,
        use_shared_memory=use_shared_memory,
    )

    if save:
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = contextlib.nullcontext()
    with save_context as directory:
        solutions_serial, paths_serial = sc.solve_many(
            device,
            parallel_method=None,
            directory=directory,
            **solve_kwargs,
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
                for s in solutions:
                    assert isinstance(s, sc.Solution)
                    assert s.solver == solver
            else:
                for lst in solutions:
                    assert isinstance(lst, list)
                for s in solutions[0]:
                    assert isinstance(s, sc.Solution)
                    assert s.solver == solver
        else:
            assert solutions is None

    if save:
        save_context.cleanup()
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = contextlib.nullcontext()
    with save_context as directory:
        with pytest.raises(ValueError):
            solutions_mp, paths_mp = sc.solve_many(
                device,
                parallel_method="mp",
                num_cpus=num_cpus,
                directory=directory,
                gpu=True,
                **solve_kwargs,
            )
        solutions_mp, paths_mp = sc.solve_many(
            device,
            parallel_method="mp",
            num_cpus=num_cpus,
            directory=directory,
            **solve_kwargs,
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
                for s in solutions:
                    assert isinstance(s, sc.Solution)
                    assert s.solver == solver
            else:
                for lst in solutions:
                    assert isinstance(lst, list)
                for s in solutions[0]:
                    assert isinstance(s, sc.Solution)
                    assert s.solver == solver
        else:
            assert solutions is None

    if save:
        save_context.cleanup()
        save_context = tempfile.TemporaryDirectory()
    else:
        save_context = contextlib.nullcontext()
    with save_context as directory:
        solutions_ray, paths_ray = sc.solve_many(
            device,
            parallel_method="ray",
            num_cpus=num_cpus,
            directory=directory,
            **solve_kwargs,
        )
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
                for s in solutions:
                    assert isinstance(s, sc.Solution)
            else:
                for lst in solutions:
                    assert isinstance(lst, list)
                for s in solutions[0]:
                    assert isinstance(s, sc.Solution)
        else:
            assert solutions is None

    if save:
        save_context.cleanup()

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
