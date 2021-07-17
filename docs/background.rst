.. superscreen

.. _background:

**********
Background
**********

The goal of ``SuperScreen`` is to model the magnetic response of a thin superconducting film,
or a structure composed of multiple superconducting films (which may or may not lie in the same plane),
to an applied inhomogeneous out-of-plane magnetic field
:math:`H_{z,\,\mathrm{applied}}(x, y, z)`.

Given :math:`H_{z,\,\mathrm{applied}}(x, y, z)` and information about the geometry and magnetic
penetration depth of all films in a superconducting structure, we aim
to calculate the thickness-integrated current density :math:`\vec{J}(x, y)`
at all points inside the films, from which one can calculate the vector magnetic field
:math:`\vec{H}(x, y, z)` at all points both inside and outside the films.

Brandt's Method
---------------

A convenient method for solving this problem was introduced in [Brandt-PRB-2005]_,
and used in [Kirtley-RSI-2016]_ and [Kirtley-SST-2016]_ to model the magnetic response
of scanning Superconducting Quantum Interference Device (SQUID) susceptometers.

Model
=====

In the London model of superconductivity, the magnetic field :math:`\vec{H}(\vec{r})`
and 3D current density :math:`\vec{j}(\vec{r})` in a superconductor with London penetration
depth :math:`\lambda(\vec{r})` obey the second London equation:
:math:`\nabla\times\vec{j}(\vec{r})=-\vec{H}(\vec{r})/\lambda^2(\vec{r})`, where
:math:`\nabla=\left(\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z}\right)`.

Brandt's model assumes that the current density :math:`\vec{j}` is approximately independent of :math:`z`,
such :math:`\vec{j}(x, y, z)\approx\vec{j}_{z_0}(x, y)` for a film lying parallel to the :math:`x-y` plane
at vertical position :math:`z_0`. Working now with the thickness-integrated current density
(or "sheet current") :math:`\vec{J}(x, y)=\vec{j}_{z_0}(x, y)\cdot d`, where :math:`d`
is the thickness of the film, the second London equation
reduces to

.. math::
    :label: eq1

    \nabla\times\vec{J}(x, y)=-\vec{H}(x, y)/\Lambda(x, y),

where :math:`\Lambda(x, y)=\lambda^2(x, y)/d` is the effective penetration depth
of the superconducting film (equal to half the Pearl length [Pearl-APL-1964]_).

.. note::

    The assumption :math:`\vec{j}(x, y, z)\approx\vec{j}_{z_0}(x, y)` is valid for
    films that are thinner than their London penetration depth (i.e. :math:`d<\lambda`),
    such that :math:`\Lambda=\lambda^2/d>\lambda`. However the model has been applied
    with some success in structures with :math:`\lambda\lesssim d`
    [Kirtley-RSI-2016]_ [Kirtley-SST-2016]_. Aside from this limitation, the method described
    below can in principle to used to model films with any effective penetration depth
    :math:`0\leq\Lambda<\infty`.

Because the sheet current has zero divergence in the superconducting film (:math:`\nabla\cdot\vec{J}=0`)
except at small contacts where current can be injected, one can express the sheet current in terms
of a scalar potential :math:`g(x, y)`, called the stream function:

.. math::
    :label: eq2

    \vec{J}(x, y) = -\hat{z}\times\nabla g
    = \nabla\times(g\hat{z})
    = \left(\frac{\partial g}{\partial y}, -\frac{\partial g}{\partial x}\right).
    
The stream function :math:`g` can be thought of as the local magnetization of the film, or the area
density of tiny dipole sources. We can re-write :eq:`eq1` for a 2D film in terms of :math:`g`:

.. math::
    :label: eq3

    \vec{H}(x, y) &= -\Lambda\left[\nabla\times\vec{J}(x, y)\right]\\
    &= -\Lambda\left[\nabla\times\left(\nabla\times(g\hat{z})\right)\right]\\
    &= -\Lambda\left[\nabla(\nabla\cdot(g\hat{z}))-\nabla^2(g\hat{z})\right]\\
    &=\Lambda\nabla^2g(x,y)\hat{z},

where :math:`\nabla^2=\nabla\cdot\nabla` is the Laplace operator. (The last line follows from the
fact that :math:`\nabla\cdot\left(g(x,y)\hat{z}\right) = 0`). From Ampere's Law, the
three components of the magnetic field at position :math:`\vec{r}=(x, y, z)` due to a
sheet of current lying in the :math:`x-y` plane (at height :math:`z'`) with stream function
:math:`g(x', y')` are given by:

.. math::
    :label: eq4

    H_x(\vec{r}) &= \int_S Q_x(\vec{r},\vec{r}')g(x', y')\,\mathrm{d}^2r'\\
    H_y(\vec{r}) &= \int_S Q_y(\vec{r},\vec{r}')g(x', y')\,\mathrm{d}^2r'\\
    H_z(\vec{r}) &= H_{z,\,\mathrm{applied}}(\vec{r})
    + \int_S Q_z(\vec{r},\vec{r}')g(x', y')\,\mathrm{d}^2r'

:math:`H_{z,\,\mathrm{applied}}(\vec{r})` is an externally-applied magnetic field (which we assume to have
no :math:`x` or :math:`y` component), :math:`S` is the film area (with :math:`g = 0` outside of the film),
and :math:`Q_x(\vec{r},\vec{r}')`, :math:`Q_y(\vec{r},\vec{r}')`, and :math:`Q_z(\vec{r},\vec{r}')`
are dipole kernel functions which give the relevant component of the magnetic field at position :math:`\vec{r}=(x, y, z)`
due to a dipole of unit strength at poition :math:`\vec{r}'=(x', y', z')`:

.. math::
    :label: eq5

    Q_x(\vec{r}, \vec{r}') &=  3\Delta z\frac{x-x'}
    {4\pi[(\Delta z)^2+\rho^2]^{5/2}}\\
    Q_y(\vec{r}, \vec{r}') &=  3\Delta z\frac{y-y'}
    {4\pi[(\Delta z)^2+\rho^2]^{5/2}}\\
    Q_z(\vec{r}, \vec{r}') &=  \frac{2(\Delta z)^2-\rho^2}
    {4\pi[(\Delta z)^2+\rho^2]^{5/2}},

where :math:`\Delta z = z - z'` and :math:`\rho=\sqrt{(x-x')^2 + (y-y')^2}`.
Comparing :eq:`eq3` and :eq:`eq4`, we have in the plane of the film:

.. math::
    :label: eq6

    \underbrace{H_z(\vec{r}) = \vec{H}(\vec{r})\cdot\hat{z}
    = \Lambda\nabla^2g(x, y)}_{z-\text{component of the total field}}
    = \underbrace{H_{z,\,\mathrm{applied}}(\vec{r})}_{\text{applied field}}
    + \underbrace{\int_S Q_z(\vec{r},\vec{r}')g(\vec{r}')\,\mathrm{d}^2r'}_{\text{screening field}},

(where now :math:`\vec{r}` and :math:`\vec{r}'` are 2D vectors, i.e. :math:`\Delta z=0`, since the film is
in the same plane as itself). From :eq:`eq6`, we arrive at an integral equation relating the stream function
:math:`g` for points inside the superconductor to the applied field :math:`H_{z,\,\mathrm{applied}}`:

.. math::
    :label: eq7

    H_{z,\,\mathrm{applied}}(\vec{r})
    = -\int_S\left[
        Q_z(\vec{r},\vec{r}')-\delta(\vec{r}-\vec{r}')\Lambda(\vec{r}')\nabla^2\right
    ]g(\vec{r}')\,\mathrm{d}^2r'

The goal, then, is to solve (invert) :eq:`eq7` for a given :math:`H_{z,\,\mathrm{applied}}`
and film geometry :math:`S` to obtain :math:`g` for all points inside the film
(with :math:`g=0` enforced outside the film). Once :math:`g(\vec{r})` is known,
the full vector magnetic field :math:`\vec{H}` can be calculated at any point :math:`\vec{r}`
from :eq:`eq4`.

Films with holes
================

In films that have holes (regions of vacuum completely surrounded by superconductor),
each hole :math:`k` can contain a trapped flux :math:`\Phi_k`, with an associated circulating
current :math:`I_{\mathrm{circ},\,k}`. The (hypothetical) applied field that would cause
such a circulating current is given by :eq:`eq7` if we set
:math:`g(\vec{r})=I_{\mathrm{circ},\,k}` for all points :math:`\vec{r}` lying inside hole :math:`k`:

.. math::
    :label: eq8

    H_{z,\,\mathrm{eff},\,k}(\vec{r}) = -\int_{\mathrm{hole}\,k}[
        Q_z(\vec{r},\vec{r}')-\Lambda(\vec{r}')\nabla^2
    ] I_{\mathrm{circ},\,k} \,\mathrm{d}^2r'.

In this case, we modify the left-hand side of :eq:`eq7` as follows:

.. math::
    :label: eq9

    H_{z,\,\mathrm{applied}}(\vec{r}) - \sum_k H_{z,\,\mathrm{eff},\,k}(\vec{r})
    = -\int_S\left[
        Q_z(\vec{r},\vec{r}')-\delta(\vec{r}-\vec{r}')\Lambda(\vec{r}')\nabla^2\right
    ]g(\vec{r}')\,\mathrm{d}^2r'

Films in multiple planes
========================

For structures with multiple films lying in different planes or "layers",
with layer :math:`\ell` lying in the plane :math:`z=z_\ell`,
the stream functions and fields for all layers can be computed self-consistently
using the following recipe:

1. Calculate the stream function :math:`g_\ell(\vec{r})` for each layer :math:`\ell` by solving
   :eq:`eq9` given an applied field :math:`H_{z,\,\mathrm{applied}}(\vec{r}, z_\ell)`.
2. For each layer :math:`\ell`, calculate the :math:`z`-component of the field due to the
   currents in all other layers :math:`k\neq\ell` (encoded in the stream function :math:`g_k(\vec{r})`)
   using :eq:`eq4`.
3. Re-solve :eq:`eq9` taking the new applied field at each layer to be the original
   applied field plus the sum of screening fields from all other layers. This is accomplished
   via the substitution:

   .. math::

        H_{z,\,\mathrm{applied}}(\vec{r}, z_\ell) \to
        H_{z,\,\mathrm{applied}}(\vec{r}, z_\ell)
        + \sum_{k\neq\ell}
        \int_S Q_z(\vec{r},\vec{r}')g_k(\vec{r}')\,\mathrm{d}^2r'.
    
4. Repeat steps 1-3 until the stream functions and fields converge.

---------------------------------------------------------------

Finite Element Implementation
-----------------------------

Here we describe the numerical implementation of the model described above.

.. .. _brandt-strategy:

.. .. figure:: /images/brandt.jpg
..    :scale: 30%
..    :align: center
..    :alt: Simulation strategy

..    Schematic of the numerical implementation of Brandt's method.


Discretized model
=================
    
In order to numerically solve :eq:`eq4` and :eq:`eq9`, we have to discretize
the films, holes, and the vacuum regions surrounding them. We use a triangular
(Delaunay) mesh, consisting of :math:`n` points (or vertices)
which together form :math:`m` triangles.

Below we denote column vectors and matrices using bold font. :math:`\mathbf{A}\mathbf{B}`
denotes matrix multiplication, with :math:`(\mathbf{A}\mathbf{B})_{ij}=\sum_{k=1}^\ell A_{ik}B_{kj}`
(:math:`\ell` being the number of columns in :math:`\mathbf{A}` and the number of
rows in :math:`\mathbf{B}`). Column vectors are treated as matrices with
:math:`\ell` rows and :math:`1` column. On the other hand, we denote element-wise
multiplication with a dot: :math:`(\mathbf{A}\cdot\mathbf{B})_{ij}=A_{ij}B_{ij}` for two matrices
and :math:`(\mathbf{A}\cdot\mathbf{v})_{ij}=A_{ij}v_{i}` for a matrix and a column vector.

The matrix version of :eq:`eq4` is:

.. math::
    :label: eq10

    \underbrace{\mathbf{h}_z}_\text{total field}
    = \underbrace{\mathbf{h}_{z,\,\mathrm{applied}}}_\text{applied field}
    + \underbrace{(\mathbf{Q}\cdot\mathbf{w})\mathbf{g}}_\text{screening field}.

The kernel matrix :math:`\mathbf{Q}` and weight matrix :math:`\mathbf{w}` together play the role of the
kernel function :math:`Q_z(\vec{r},\vec{r}')` for all points lying in the plane of the film.
They are both :math:`n\times n` matrices and are determined by the geometry of the films.
:math:`\mathbf{h}_z`, :math:`\mathbf{h}_{z,\,\mathrm{applied}}`, and :math:`\mathbf{g}` are all
:math:`n\times 1` vectors, with each row representing the value of the quantity at the
corresponding vertex in the mesh. There are several different methods for constructing the
weight matrix :math:`\mathbf{w}`, which are discussed :ref:`below <weight-matrix>`. The kernel
matrix :math:`\mathbf{Q}` is defined in terms of a matrix with
:math:`(\mathbf{q})_{ij} = \left(4\pi|\vec{r}_i-\vec{r}_j|^3\right)^{-1}`
(which is :math:`\lim_{\Delta z\to 0}Q_z(\vec{r},\vec{r}')` cf. :eq:`eq5`),
and a vector :math:`\mathbf{C}`:

.. math::
    :label: eq11

    Q_{ij} = (\delta_{ij}-1)q_{ij}
    + \delta_{ij}\frac{1}{w_{ij}}\left(C_i + \sum_{l\neq i}q_{il}w_{il}\right),

where :math:`\delta_{ij}` is the Kronecker delta function. The diagonal terms involving
:math:`\mathbf{C}` are meant to work around the fact that :math:`(\mathbf{q})_{ii}` diverge
(see [Brandt-PRB-2005]_ for more details), and the vector is defined as

.. math::
    :label: eq12

    C_i = \frac{1}{4\pi}\sum_{p,q=\pm1}\sqrt{[\Delta x - p(x_i-\bar{x})]^{-2} + [\Delta y - q(y_i-\bar{y})]^{-2}},

where :math:`\Delta x=(x_\mathrm{max}-x_\mathrm{min})/2` and :math:`\Delta y=(y_\mathrm{max}-y_\mathrm{min})/2`
are half the side lengths of a rectangle bounding the modeled film(s) and :math:`(\bar{x}, \bar{y})` are the
coordinates of the center of the rectangle. The matrix version of :eq:`eq9`
is:

.. math::
    :label: eq13

    -(\mathbf{Q}\cdot\mathbf{w}-\mathbf{\Lambda}\cdot\mathbf{\nabla}^2)\mathbf{g}
    = \mathbf{h}_{z,\,\mathrm{applied}} - \sum_{\mathrm{holes}\,k}\mathbf{h}_{z,\,\mathrm{eff},\,k}

(where we exclude points in the mesh lying outside of the superconducting film, but keep points
inside holes in the film). :math:`\mathbf{\Lambda}` is either a scalar or a vector defining the
effective penetration depth at every included vertex in the mesh, and :math:`\mathbf{\nabla}^2`
is the :ref:`Laplace operator <laplace-operator>`, an :math:`n\times n` matrix defined such that
:math:`\mathbf{\nabla}^2\mathbf{f}` computes the Laplacian :math:`\nabla^2f(x,y)` of a
function :math:`f(x,y)` defined on the mesh.

:eq:`eq13` is a matrix equation relating the applied field to the stream function
inside a superconducting film, which can efficiently be solved (e.g. by matrix inversion)
for the unknown vector :math:`\mathbf{g}`, the stream function inside the film. Since the stream
function outside the film and inside holes in the film is already known, solving :eq:`eq13`
gives us the stream function for the full mesh:

.. math::
    :label: eq14

    \mathbf{g} = \begin{cases}
        \left(-[\mathbf{Q}\cdot\mathbf{w}-\mathbf{\Lambda}\cdot\mathbf{\nabla}^2]\right)^{-1}
        \left(\mathbf{h}_{z,\,\mathrm{applied}} - \sum_{\mathrm{holes}\,k}\mathbf{h}_{z,\,\mathrm{eff},\,k}\right)
            & \text{inside the film}\\
        I_{\mathrm{circ},\,k}
            & \text{inside hole }k\\
        0
            & \text{elsewhere}
    \end{cases}

Once the stream function :math:`\mathbf{g}` is known for the full mesh,
the sheet current flowing in the film can be computed from :eq:`eq2`,
the :math:`z`-component of the total field at the plane of the film can be computed
from :eq:`eq10`, and the full vector magnetic field :math:`\vec{H}(x, y, z)`
at any point in space can be computed from :eq:`eq4` (and its analogs for
the :math:`x` and :math:`y` components of the field).

.. _laplace-operator:

Laplace operator
================

The definition of the mesh Laplace operator :math:`\mathbf{\nabla}^2` (also called the
Laplace-Beltrami operator [Laplacian-SGP-2014]_) deserves special attention, as it reduces
the problem of solving a partial differential equation :math:`\nabla^2g(x,y)=f(x,y)` to the
numerically-tractable problem of solving a matrix equation
:math:`\mathbf{\nabla}^2\mathbf{g}=\mathbf{f}`. As described in [Vaillant-Laplacian-2013]_
and [Laplacian-SGP-2014]_ the Laplace operator :math:`\mathbf{\nabla}^2` for a mesh is
defined in terms of two matrices, the mass matrix :math:`\mathbf{M}` and the
Laplacian matrix :math:`\mathbf{L}`: :math:`\mathbf{\nabla}^2 = \mathbf{M}^{-1}\mathbf{L}`.

Mass matrix
***********

The mass matrix gives an effective area to each vertex in the mesh. There are multiple
ways to construct the mass matrix, but here we use a "lumped" mass matrix, which is diagonal
with elements :math:`(\mathbf{M})_{ii} = \sum_{t\in\mathcal{N}(i)}\frac{1}{3}\mathrm{area}(t)`,
where :math:`\mathcal{N}(i)` is the set of triangles :math:`t` adjacent to vertex :math:`i`.
(See :ref:`image below <img-weights>`, where :math:`(\mathbf{M})_{ii} = A_i`.
Image reference: [Vaillant-Laplacian-2013]_.)

.. _img-weights:

.. image:: /images/cotan_angles_.png
    :align: center
    :width: 30%

.. _weight-matrix:

Weight matrix
*************

Each element :math:`w_{ij}` of the symmetric weight matrix :math:`\mathbf{w}` assigns a weight
to the edge connecting vertex :math:`i` and vertex :math:`j` in the mesh.
We use a normalized version of the weight matrix, where the sum of all off-diagonal
elements in each row (or column) is :math:`1`, and each diagonal element is :math:`1`.
Thus we can write :math:`\mathbf{w}` in terms of an unnormalized weight matrix :math:`\mathbf{W}`:

.. math::

    w_{ij} = \begin{cases}
        1&\text{if }i = j\\
        W_{ij} / \sum_{i\neq j} W_{ij}&\text{otherwise}
    \end{cases}

There are several different methods for constructing the unnormalized weight matrix
:math:`\mathbf{W}`:

1. Uniform weighting: In this case, :math:`\mathbf{W}` is simply the
   `adjacency matrix <https://mathworld.wolfram.com/AdjacencyMatrix.html>`_ for the mesh.
   
    .. math::

        W_{ij} =
        \begin{cases}
            0&\text{if }i=j\\
            1&\text{if }i\text{ is adjacent to }j\\
            0&\text{otherwise}
        \end{cases}

2. Inverse-Euclidean weighting: Each edge is weighted by the inverse of its length:
   :math:`|\vec{r}_i-\vec{r}_j|^{-1}`, where :math:`\vec{r}_i` is the position of vertex :math:`i`.

    .. math::

        W_{ij} =
        \begin{cases}
            0&\text{if }i=j\\
            |\vec{r}_i-\vec{r}_j|^{-1}&\text{if }i\text{ is adjacent to }j\\
            0&\text{otherwise}
        \end{cases}

3. Half-cotangent weighting: Each edge is weighted by the half the sum of the cotangents of the
   two angles opposite to it. See :ref:`image above <img-weights>`.
   Image reference: [Vaillant-Laplacian-2013]_.).

    .. math::

        W_{ij} =
        \begin{cases}
            0&\text{if }i=j\\
            \frac{1}{2}\left(\cot\alpha_{ij}+\cot\beta_{ij}\right)&\text{if }i\text{ is adjacent to }j\\
            0&\text{otherwise}
        \end{cases}


Laplacian matrix
****************

The Laplacian matrix :math:`\mathbf{L}` is defined in terms of the weight matrix :math:`\mathbf{w}`:

.. math::

    (\mathbf{L})_{ij} = (\mathbf{w})_{ij} - \delta_{ij}\sum_{\ell}w_{i\ell}.

Finally, the Laplace operator is given by:

.. math::

    \mathbf{\nabla}^2 = \mathbf{M}^{-1}\mathbf{L}.

.. toctree::
    :numbered:
