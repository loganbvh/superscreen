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
depth :math:`\lambda` obey the second London equation:
:math:`\nabla\times\vec{j}(\vec{r})=-\vec{H}(\vec{r})/\lambda^2`, where
:math:`\nabla=\left(\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \frac{\partial}{\partial z}\right)`.

Brandt's model assumes that the current density :math:`\vec{j}` is approximately independent of :math:`z`,
such :math:`\vec{j}(x, y, z)\approx\vec{j}_{z_0}(x, y)` for a film lying parallel to the :math:`x-y` plane
at vertical position :math:`z_0`. Working now with the thickness-integrated current density
(or "sheet current") :math:`\vec{J}(x, y)=\vec{j}_{z_0}(x, y)\cdot d`, where :math:`d`
is the thickness of the film, the second London equation
reduces to

.. math::
    :label: eq1

    \nabla\times\vec{J}(x, y)=-\vec{H}(x, y)/\Lambda,

where :math:`\Lambda=\lambda^2/d` is the effective penetration depth
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
        Q_z(\vec{r},\vec{r}')-\delta(\vec{r}-\vec{r}')\Lambda\nabla^2\right
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
        Q_z(\vec{r},\vec{r}')-\delta(\vec{r}-\vec{r}')\Lambda\nabla^2
    ] I_{\mathrm{circ},\,k} \,\mathrm{d}^2r'.

In this case, we modify the left-hand side of :eq:`eq7` as follows:

.. math::
    :label: eq9

    H_{z,\,\mathrm{applied}}(\vec{r}) - \sum_k H_{z,\,\mathrm{eff},\,k}(\vec{r})
    = -\int_S\left[
        Q_z(\vec{r},\vec{r}')-\delta(\vec{r}-\vec{r}')\Lambda\nabla^2\right
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


Applied bias currents
=====================

For devcies composed of a single film in a single layer, a bias current can be applied through one or more terminals or current contacts.
See `Terminal currents <notebooks/terminal-currents.ipynb>`_ for more details.

---------------------------------------------------------------

Finite Element Implementation
-----------------------------

Here we describe the numerical implementation of the model outlined above.

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
multiplication with a lower dot: :math:`(\mathbf{A}.\mathbf{B})_{ij}=A_{ij}B_{ij}`.
:math:`\mathbf{A}^T` denotes the transpose of matrix :math:`\mathbf{A}`.

The matrix version of :eq:`eq4` is:

.. math::
    :label: eq10

    \underbrace{\mathbf{h}_z}_\text{total field}
    = \underbrace{\mathbf{h}_{z,\,\mathrm{applied}}}_\text{applied field}
    + \underbrace{(\mathbf{Q}.\mathbf{w}^T)\mathbf{g}}_\text{screening field}.

The :math:`n\times n` kernel matrix :math:`\mathbf{Q}` represents the kernel function :math:`Q_z(\vec{r},\vec{r}')`
for all points lying in the plane of the film, and the :math:`n\times 1` weight vector :math:`\mathbf{w}`,
which assigns an effective area to each vertex in the mesh, represents the differential element
:math:`\mathrm{d}^2r'`. Both :math:`\mathbf{Q}` and :math:`\mathbf{w}` are solely determined by the
geometry of the mesh, so they only need to be computed once for a given device.
:math:`\mathbf{h}_z`, :math:`\mathbf{h}_{z,\,\mathrm{applied}}`, and :math:`\mathbf{g}` are all
:math:`n\times 1` vectors, with each row representing the value of the quantity at the
corresponding vertex in the mesh. The vector :math:`\mathbf{w}` is equal to the diagonal of
the "lumped mass matrix" :math:`\mathbf{M}`:
:math:`w_i=M_{ii} = \frac{1}{3}\sum_{t\in\mathcal{N}(i)}\mathrm{area}(t)`,
where :math:`\mathcal{N}(i)` is the set of triangles :math:`t` adjacent to vertex :math:`i`.
The kernel matrix :math:`\mathbf{Q}` is defined in terms of a matrix with
:math:`(\mathbf{q})_{ij} = \left(4\pi|\vec{r}_i-\vec{r}_j|^3\right)^{-1}`
(which is :math:`\lim_{\Delta z\to 0}Q_z(\vec{r},\vec{r}')` cf. :eq:`eq5`),
and a vector :math:`\mathbf{C}`:

.. math::
    :label: eq11

    Q_{ij} = (\delta_{ij}-1)q_{ij}
    + \delta_{ij}\frac{1}{w_{j}}\left(C_i + \sum_{l\neq i}q_{il}w_{l}\right),

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

    -(\mathbf{Q}.\mathbf{w}^T-\Lambda\mathbf{\nabla}^2)\mathbf{g}
    = \mathbf{h}_{z,\,\mathrm{applied}} - \sum_{\mathrm{holes}\,k}\mathbf{h}_{z,\,\mathrm{eff},\,k}

(where we exclude points in the mesh lying outside of the superconducting film, but keep points
inside holes in the film). :math:`\mathbf{\nabla}^2` is the :ref:`Laplace operator <laplace-operator>`,
an :math:`n\times n` matrix defined such that
:math:`\mathbf{\nabla}^2\mathbf{f}` computes the Laplacian :math:`\nabla^2f(x,y)` of a
function :math:`f(x,y)` defined on the mesh.

:eq:`eq13` is a matrix equation relating the applied field to the stream function
inside a superconducting film, which can efficiently be solved (e.g. by matrix inversion)
for the unknown vector :math:`\mathbf{g}`, the stream function inside the film. Since the stream
function outside the film and inside holes in the film is already known, solving :eq:`eq13`
gives us the stream function for the full mesh. Defining
:math:`\mathbf{K} = \left(\mathbf{Q}\cdot\mathbf{w}^T-\Lambda\mathbf{\nabla}^2\right)^{-1}`, we have


.. math::
    :label: eq14

    \mathbf{g} = \begin{cases}
        -\mathbf{K}
        \left(\mathbf{h}_{z,\,\mathrm{applied}} - \sum_{\mathrm{holes}\,k}\mathbf{h}_{z,\,\mathrm{eff},\,k}\right)
            & \text{inside the film}\\
        I_{\mathrm{circ},\,k}
            & \text{inside hole }k\\
        0
            & \text{elsewhere}
    \end{cases}

If there is a vortex containing flux :math:`\Phi_j` trapped in a film at position :math:`\vec{r}_j` indexed as mesh vertex :math:`j`,
then for each position :math:`\vec{r}_i` within that film, we add to the stream function :math:`g_i` the quantity :math:`\mu_0^{-1}\Phi_jK_{ij} / w_{j}`, 
here :math:`K_{ij}` is an element of the inverse matrix defined above, and :math:`w_{j}` is an element of the weight matrix which assigns an effective area to the mesh vertex at which the vortex is located. This process amounts to numerically inverting :eq:`eq9` in the presence of delta-function magnetic sources representing trapped vortices, as described in [Brandt-PRB-2005]_.

Once the stream function :math:`\mathbf{g}` is known for the full mesh,
the sheet current flowing in the film can be computed from :eq:`eq2`,
the :math:`z`-component of the total field at the plane of the film can be computed
from :eq:`eq10`, and the full vector magnetic field :math:`\vec{H}(x, y, z)`
at any point in space can be computed from :eq:`eq4` (and its analogs for
the :math:`x` and :math:`y` components of the field).

.. _laplace-operator:

Laplace and gradient operators
==============================

The definitions of the Laplace operator :math:`\mathbf{\nabla}^2` (also called the Laplace-Beltrami operator)
and the gradient operator :math:`\vec{\nabla}=(\nabla_x, \nabla_y)^T` deserve special attention, as these two
operators reduce the problem of solving a partial differential equation to the problem of solving a
matrix equation [Laplacian-SGP-2014]_.  Given a mesh consisting of :math:`n` vertices and :math:`m` triangles,
and a scalar field :math:`f(x, y)` represented by an :math:`n\times 1` vector :math:`\mathbf{f}` containing the values
of the field at the mesh vertices, the goal is to construct matrices :math:`\nabla^2` and :math:`\vec{\nabla}=(\nabla_x, \nabla_y)^T`
such that the matrix products :math:`\nabla^2\mathbf{f}` and :math:`\vec{\nabla}\mathbf{f}` approximate the Laplacian
:math:`\left(\frac{\partial^2f}{\partial x^2}+\frac{\partial^2f}{\partial y^2}\right)` and the gradient
:math:`\left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)` of :math:`f(x, y)` at the mesh vertices.

As described in [Vaillant-Laplacian-2013]_
and [Laplacian-SGP-2014]_ the Laplace operator :math:`\mathbf{\nabla}^2` for a mesh is
defined in terms of two matrices, the mass matrix :math:`\mathbf{M}` and the
Laplacian matrix :math:`\mathbf{L}`: :math:`\mathbf{\nabla}^2 = \mathbf{M}^{-1}\mathbf{L}`
The mass matrix gives an effective area to each vertex in the mesh. There are multiple
ways to construct the mass matrix, but here we use a "lumped" mass matrix, which is diagonal
with elements :math:`(\mathbf{M})_{ii} = \sum_{t\in\mathcal{N}(i)}\frac{1}{3}\mathrm{area}(t)`,
where :math:`\mathcal{N}(i)` is the set of triangles :math:`t` adjacent to vertex :math:`i`.
(See :ref:`image below <img-weights>`, where :math:`(\mathbf{M})_{ii} = A_i`.
Image reference: [Vaillant-Laplacian-2013]_.) The vector :math:`\mathbf{w}` discussed
above is simply the diagonal of the mass matrix :math:`\mathbf{M}`.

.. _img-weights:

.. image:: /images/cotan_angles_.png
    :align: center
    :width: 30%

The Laplacian matrix :math:`\mathbf{L}` is defined in terms of the weight matrix :math:`\mathbf{W}`:

.. math::

    (\mathbf{L})_{ij} = (\mathbf{W})_{ij} - \delta_{ij}\sum_{\ell}W_{i\ell}.

There are several different methods for constructing the weight matrix
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

Finally, the Laplace operator is given by:

.. math::

    \mathbf{\nabla}^2 = \mathbf{M}^{-1}\mathbf{L}.

By default, ``SuperScreen`` uses half-cotangent weighting.

We construct the two :math:`n\times n` gradient matrices :math:`\nabla_x` and :math:`\nabla_y`, using the
"average gradient on a star" (or AGS) approach [Mancinelli-STAG-2019]_. Briefly, we first construct two
:math:`m\times n` matrices, :math:`\vec{\nabla}_t=(\nabla_{t,x}, \nabla_{t,y})^T` using "per-cell linear estimation"
(or PCE) [Mancinelli-STAG-2019]_, where :math:`\nabla_{t,x}\mathbf{f}` maps the field values at the vertices :math:`\mathbf{f}`
to an estimate of the :math:`x`-component of the gradient at the triangle centroids (centers-of-mass).
The matrices :math:`\nabla_x` and :math:`\nabla_y` are then computed by, for each vertex :math:`i`,
taking the weighted average of :math:`\nabla_{t,x}` and :math:`\nabla_{t,y}` over adjacent triangles :math:`t\in\mathcal{N}(i)`,
with weights given by the angle between the two sides of the triangle adjacent to the vertex.
The resulting :math:`\vec{\nabla}=(\nabla_x, \nabla_y)^T` is a :math:`2\times n\times n` stack of matrices defined such that
:math:`\vec{\nabla}\mathbf{f}` produces a :math:`2\times n` matrix representing the gradient of :math:`f(x, y)` at the mesh vertices,
with the first and second rows of :math:`\vec{\nabla}\mathbf{f}` containing the :math:`x` and :math:`y` components of the gradient, respectively.


.. _inhomogeneous:

Inhomogeneous Films
-------------------

The London equation (:eq:`eq1`) is valid only under the assumption that the London penetration depth :math:`\lambda`, a proxy for the superfluid density, is constant as a function of position. In cases where the superfluid density varies as a function of position, Ginzburg-Landau theory provides a more accurate description of the magnetic response of the system. Nevertheless, in an effort to model inhomogeneous superconductors using London theory, one can write out the "inhomogeneous second London equation" for a superconductor with spatially-varying London penetration depth :math:`\lambda(\vec{r})` ([Cave-Evetts-LTP-1986]_, [Kogan-Kirtley-PRB-2011]_):

.. math::
    :label: eq15

    \vec{H}(\vec{r})&=-\vec{\nabla}\times\left(\lambda^2(\vec{r})\vec{j}(\vec{r})\right)\\
    &=-\left(\lambda^2(\vec{r})\vec{\nabla}\times\vec{j}(\vec{r}) + \vec{\nabla}\lambda^2(\vec{r})\times\vec{j}(\vec{r})\right).


In the 2D limit, i.e. a thin film with thickness :math:`d\ll\lambda(x, y)` lying parallel to the :math:`x-y` plane carrying sheet current density :math:`\vec{J}(x, y)=\vec{j}(\vec{r})\cdot d` we have:

.. math::
    :label: eq16

    \vec{H}(x, y)&=-\vec{\nabla}\times(\Lambda\vec{J})\\
    &=-\left(\Lambda\vec{\nabla}\times\vec{J}+\vec{\nabla}\Lambda\times\vec{J}\,\right)\\
    &=\left(\Lambda\nabla^2g+\vec{\nabla}\Lambda\cdot\vec{\nabla}g\right)\hat{z},

where :math:`\Lambda=\Lambda(x, y)`, :math:`g=g(x, y)`, :math:`\vec{\nabla}=\left(\frac{\partial}{\partial x},\frac{\partial}{\partial y}\right)`,
and :math:`\vec{J}=\vec{J}(x, y)=\vec{\nabla}\times(g\hat{z})`.

If one defines an inhomogeneous effective penetration depth :math:`\Lambda(x, y)` in a ``SuperScreen`` model, :eq:`eq16`, rather than :eq:`eq1`, is solved numerically as follows. For a mesh with :math:`n` vertices, the effective penetration depth is represented by an :math:`n\times 1` vector :math:`\mathbf{\Lambda}`. :eq:`eq13` and :eq:`eq14` are updated according to:

.. math::

    &\mathbf{Q}.\mathbf{w}^T-\Lambda\mathbf{\nabla}^2\to\\
    &\mathbf{Q}.\mathbf{w}^T-\mathbf{\Lambda}^T.\mathbf{\nabla}^2-\vec{\nabla}\mathbf{\Lambda}\cdot\vec{\nabla}

The notation :math:`\vec{\nabla}\mathbf{f}\cdot\vec{\nabla}` indicates an inner (dot) product over the two spatial dimensions, resulting in an :math:`n\times n` matrix such that :math:`(\vec{\nabla}\mathbf{f}\cdot\vec{\nabla})\mathbf{g}` computes :math:`(\vec{\nabla}f(x, y))\cdot(\vec{\nabla}g(x, y))` (see :ref:`Laplace and gradient operators <laplace-operator>`).

Note that, unlike :eq:`eq1` in which :math:`\Lambda` is assumed to be constant as a function of position :math:`\vec{r}`, solutions to :eq:`eq16` do not necessarily satisfy the fluxoid quantization condition :math:`\Phi^f_S=0` for simply-connected superconducting regions :math:`S` in which :math:`\vec{\nabla}\Lambda(\vec{r})\neq0` , where

.. math::
	\Phi^f_S=\underbrace{\int_S\mu_0H_z(\vec{r})\,\mathrm{d}^2r}_{\text{flux part}} + \underbrace{\oint_S\mu_0\Lambda(\vec{r})\vec{J}(\vec{r})\cdot\mathrm{d}\vec{r}}_{\text{supercurrent part}}.

.. toctree::
    :numbered:
