.. superscreen

************
Installation
************

.. role:: bash(code)
   :language: bash

.. role:: python(code)
  :language: python

``SuperScreen`` requires ``Python >=3.8, <3.12`` and can be installed either from
`PyPI <https://pypi.org/project/superscreen/>`_, the Python Package index,
or from the ``SuperScreen`` `GitHub repository <https://github.com/loganbvh/superscreen>`_.

We recommend creating a new
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
for ``SuperScreen`` to avoid dependency conflicts with other packages. To create a new conda environment called
``superscreen``, run ``conda create --name superscreen python="3.x"``, where ``x`` is one of ``{8, 9, 10, 11}``.
After the environment has been created, run ``conda activate superscreen`` to activate it.


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
