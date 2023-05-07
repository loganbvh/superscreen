import os

import h5py
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
        sc.Layer("layer1", london_lambda=0.5, thickness=0.05, z0=1.0),
        sc.Layer("layer2", london_lambda=0.1, thickness=0.05, z0=2.0),
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
    device.make_mesh(max_edge_length=0.5, min_points=1500, smooth=100)
    return device


def test_ray_initialized(ray_initialized):
    assert ray_initialized


@pytest.mark.parametrize(
    "vortices, return_solutions, save, use_shared_memory",
    [
        (None, True, True, True),
        (sc.Vortex(x=0, y=0, film="disk"), True, False, True),
        ([sc.Vortex(x=0, y=0, film="disk")], False, True, False),
        # (
        #     [
        #         [sc.Vortex(x=0, y=0, film="disk")],
        #         [
        #             sc.Vortex(x=0, y=0, film="disk"),
        #             sc.Vortex(x=1, y=-1, film="disk"),
        #         ],
        #     ],
        #     True,
        #     False,
        #     True,
        # ),
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
    tmp_path,
):
    applied_field = sc.sources.ConstantField(0)

    circulating_currents = [{"ring1_hole": f"{i} uA"} for i in range(2)]

    solve_kwargs = dict(
        applied_fields=applied_field,
        circulating_currents=circulating_currents,
        vortices=vortices,
        iterations=5,
        return_solutions=return_solutions,
        keep_only_final_solution=keep_only_final_solution,
        use_shared_memory=use_shared_memory,
    )

    # Serial
    save_path = tmp_path / "serial.h5" if save else None
    solutions_serial, path_serial = sc.solve_many(
        device,
        parallel_method="serial",
        save_path=save_path,
        **solve_kwargs,
    )
    solutions, path = solutions_serial, path_serial
    assert path == save_path
    if save:
        if keep_only_final_solution:
            loaded_solutions = sc.Solution.load_solutions(save_path)
        else:
            loaded_solutions = []
            for i in range(len(circulating_currents)):
                with h5py.File(save_path, "r") as h5file:
                    loaded_solutions.append(sc.Solution.load_solutions(h5file[str(i)]))

    if return_solutions:
        assert isinstance(solutions, list)
        assert len(solutions) == len(circulating_currents)
        if save:
            assert solutions == loaded_solutions
    else:
        assert solutions is None

    # Multiprocessing
    save_path = tmp_path / "mp.h5" if save else None
    solutions_mp, path_mp = sc.solve_many(
        device,
        parallel_method="mp",
        num_cpus=num_cpus,
        save_path=save_path,
        **solve_kwargs,
    )
    solutions, path = solutions_mp, path_mp
    assert path == save_path
    if save:
        if keep_only_final_solution:
            loaded_solutions = sc.Solution.load_solutions(save_path)
        else:
            loaded_solutions = []
            for i in range(len(circulating_currents)):
                with h5py.File(save_path, "r") as h5file:
                    loaded_solutions.append(sc.Solution.load_solutions(h5file[str(i)]))

    if return_solutions:
        assert isinstance(solutions, list)
        assert len(solutions) == len(circulating_currents)
        if save:
            assert solutions == loaded_solutions
    else:
        assert solutions is None

    # Ray
    save_path = tmp_path / "ray.h5" if save else None
    solutions_ray, path_ray = sc.solve_many(
        device,
        parallel_method="ray",
        num_cpus=num_cpus,
        save_path=save_path,
        **solve_kwargs,
    )
    solutions, path = solutions_ray, path_ray
    assert path == save_path
    if save:
        if keep_only_final_solution:
            loaded_solutions = sc.Solution.load_solutions(save_path)
        else:
            loaded_solutions = []
            for i in range(len(circulating_currents)):
                with h5py.File(save_path, "r") as h5file:
                    loaded_solutions.append(sc.Solution.load_solutions(h5file[str(i)]))

    if return_solutions:
        assert isinstance(solutions, list)
        assert len(solutions) == len(circulating_currents)
        if save:
            assert solutions == loaded_solutions
    else:
        assert solutions is None

    if return_solutions:
        if keep_only_final_solution:
            for sol_serial, sol_mp, sol_ray in zip(
                solutions_serial, solutions_mp, solutions_ray
            ):
                eq = [
                    sol_serial.equals(sol_mp),
                    sol_serial.equals(sol_ray),
                    sol_ray.equals(sol_mp),
                ]
                assert all(eq)
        else:
            for solutions in zip(solutions_serial, solutions_mp, solutions_ray):
                for sol_serial, sol_mp, sol_ray in zip(*solutions):
                    eq = [
                        sol_serial.equals(sol_mp),
                        sol_serial.equals(sol_ray),
                        sol_ray.equals(sol_mp),
                    ]
                    assert all(eq)
