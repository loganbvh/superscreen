.. superscreen

.. _gotchas:

*********************
Common issues/gotchas
*********************

Below we list common issues and gotchas encountered when using ``SuperScreen``, along with the recommended solutions.
If you encounter an issue not listed here, or the recommended solution does not work for you, please
`open an issue on GitHub <https://github.com/loganbvh/superscreen/issues>`_.

- ``Device.make_mesh()`` hangs for a :class:`superscreen.Device` with transport terminals.

    .. tip::
        In order to correctly set the boundary conditions for films with transport terminals
        (see `Terminal currents <notebooks/terminal-currents.ipynb>`_), ``SuperScreen`` generates
        a mesh for such films where the boundary vertices of the mesh are exactly the same as the
        boundary vertices of the associated :class:`superscreen.Polygon`. As a result, ``Device.make_mesh()``
        can hang if the distance between vertices in the film :class:`superscreen.Polygon` is greater than
        the ``max_edge_length`` requested in ``Device.make_mesh()``. To fix this, simply increase the number
        of points in the :class:`superscreen.Polygon` defining the film. For example, to double the number of points
        in the polygon, run:

        .. code-block:: python

            polygon.points = polygon.resample(2 * len(polygon.points))

  

- Poor performance when running ``SuperScreen`` in multiple Python processes.

    .. tip::    
        ``SuperScreen`` uses `numba <https://numba.pydata.org/>`_ to automatically perform some computations in parallel
        across multiple CPUs. To avoid competition between multiple Python processes running ``SuperScreen``,
        you can set the number of threads available to ``numba`` in each process. For example, if your computer
        has 8 physical CPU cores and you are running ``SuperScreen`` in 2 different Python processes,
        you should tell ``numba`` to use 8 / 2 = 4 threads in each Python process.

        .. code-block:: python

            import joblib
            import numba

            # Number of Python processes in which you will run SuperScreen
            number_of_python_processes = 2
            # Number of physical CPU cores available
            physical_cpus = joblib.cpu_count(only_physical_cores=True)
            # Tell numba how many threads to use in each Python process
            numba.set_num_threads(int(physical_cpus / number_of_python_processes))

        For more details, see `Setting the Number of Threads <https://numba.pydata.org/numba-doc/latest/user/threading-layer.html#setting-the-number-of-threads>`_
        in the ``numba`` documentation.

        A similar problem can occur when running ``SuperScreen`` on a cluster using a scheduler such as
        `Slurm <https://slurm.schedmd.com/documentation.html>`_. ``numba`` sets the default number of threads
        according to ``multiprocessing.cpu_count()``, which does not know about the number of CPUs requested
        from the scheduler. On the other hand ``joblib.cpu_count()`` does know about the number of CPUs
        allocated by the scheduler. Therefore, when running ``SuperScreen`` on such a cluster you should always run

        .. code-block:: python

            import joblib
            import numba

            numba.set_num_threads(joblib.cpu_count(only_physical_cores=True))

        or set the environment variable ``NUMBA_NUM_THREADS`` prior to importing ``numba``/``superscreen``.



