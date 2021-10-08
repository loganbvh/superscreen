.. superscreen

************
Installation
************

``SuperScreen`` requires ``Python >=3.6, <3.10`` and can only be installed from source.
In the future, the package will be added to `PyPi <https://pypi.org/>`_, the Python Package index,
making it ``pip``-installable.

We recommend creating a new
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for ``SuperScreen`` to avoid dependency conflicts with other packages. To create a new conda environment called
``superscreen``, run ``conda create --name superscreen python=3.x``, where ``x`` is one of ``{6, 7, 8, 9}``.
After the environment has been created, run ``conda activate superscreen`` to activate it.


.. important::

  - On **Windows**, ensure that the latest
    `Microsoft Visual C++ runtime
    <https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0>`_
    is installed before installing ``superscreen``.
  - On **MacOS**, ensure that `Xcode command line tools <https://mac.install.guide/commandlinetools/>`_
    are installed before installing ``superscreen``.


Install from source
-------------------

- `Clone <https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository-from-github/cloning-a-repository>`_
  or download the `superscreen repository <https://github.com/loganbvh/superscreen>`_ on GitHub
- From the base directory of the repository, run
  
   .. code-block:: bash

      pip install -e .

Install with pip
----------------

- Coming soon...


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
