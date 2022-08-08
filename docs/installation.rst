.. superscreen

************
Installation
************

.. role:: bash(code)
   :language: bash

.. role:: python(code)
  :language: python

``SuperScreen`` requires ``Python >=3.7, <3.10`` and can be installed either from
`PyPI <https://pypi.org/project/superscreen/>`_, the Python Package index,
or from the ``SuperScreen`` `GitHub repository <https://github.com/loganbvh/superscreen>`_.

We recommend creating a new
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for ``SuperScreen`` to avoid dependency conflicts with other packages. To create a new conda environment called
``superscreen``, run ``conda create --name superscreen python=3.x``, where ``x`` is one of ``{7, 8, 9}``.
After the environment has been created, run ``conda activate superscreen`` to activate it.


.. important::

  - On **Windows**, ensure that the latest
    `Microsoft Visual C++ runtime
    <https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0>`_
    is installed before installing ``superscreen``.
  - On **MacOS**, ensure that `Xcode command line tools <https://mac.install.guide/commandlinetools/>`_
    are installed before installing ``superscreen``.

Install via ``pip``
-------------------

From `PyPI <https://pypi.org/project/superscreen/>`_, the Python Package Index:

.. code-block:: bash

  pip install superscreen

From the `SuperScreen GitHub repository <https://github.com/loganbvh/superscreen/>`_:

.. code-block:: bash

  pip install git+https://github.com/loganbvh/superscreen.git

Developer Installation
----------------------

To install an editable version of ``SuperScreen``, run:

.. code-block:: bash

  git clone https://github.com/loganbvh/superscreen.git
  cd superscreen
  pip install -e .


Verify the installation
-----------------------

To verify your installation by running the ``superscreen`` test suite,
execute the following commands in a Python session:

.. code-block:: python

    >>> import superscreen.testing as st
    >>> st.run()

If you prefer, you can also run the ``superscreen`` tests in a single line:

.. code-block:: bash

    python -m superscreen.testing

GPU acceleration
----------------

``SuperScreen`` can use a graphics processing unit (GPU) as a hardware accelerator to speed up computations.
GPU acceleration relies on the `JAX library <https://github.com/google/jax>`_  from Google and requires
a machine running Linux (or `Windows Subsystem Linux, WSL <https://docs.microsoft.com/en-us/windows/wsl/about>`_)
with an Nvidia GPU.

.. note::

    These installation instructions require that you have installed ``superscreen`` in a conda environment
    as described above. Below we assume that this conda environment is called ``superscreen``.

CUDA and JAX can be installed as follows:

.. code-block:: bash

  # Activate superscreen conda environment
  conda activate superscreen
  # Install CUDA from the Nvidia conda channel:
  # https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation
  conda install cuda -c nvidia
  # Install JAX based on instructions provided by Google:
  # https://github.com/google/jax#installation
  pip install --upgrade pip
  pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Once installed, running ``SuperScreen`` on a GPU is as simple as passing the keyword argument :python:`gpu=True` to
:python:`superscreen.solve()`. See `GPU acceleration <notebooks/gpu.ipynb>`_ for an demonstration of
GPU-accelerated ``SuperScreen`` simulations.

.. note::

  To ensure that JAX can find the appropriate GPU libraries, you may need to update the environment variable
  :bash:`export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}` after activating your
  conda environment and before starting Python. Alternatively, in your Python session you can run:

  .. code-block:: python

    >>> import os
    >>> os.environ["LD_LIBRARY_PATH"] = f"$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

  If you don't set this environment variable, you may see a CUDA-related error such as ``cuSolver internal error``
  when you try to run :python:`superscreen.solve(..., gpu=True)`. ``$CONDA_PREFIX`` is an environment variable
  set automatically by conda that points to the directory of the active conda environment.
  For example, it might be ``$HOME/opt/anaconda3/envs/superscreen``.
