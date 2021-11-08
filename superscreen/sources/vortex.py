from typing import Union

import numpy as np

from ..parameter import Parameter


def monopole(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    x0: float = 0,
    y0: float = 0,
    z0: float = 0,
    nPhi0: Union[int, float] = 1,
) -> Union[float, np.ndarray]:
    """Field :math:`\\mu_0H_z` from an isolated (monopole)
    in units of ``Phi_0 / (length_units)**2``.

    .. math::

        \\mu_0H_z(\\vec{r}-\\vec{r}_0) = \\frac{n\\Phi_0}{2\\pi}
            \\frac{(\\vec{r}-\\vec{r}_0)\\cdot\\hat{z}}{|(\\vec{r}-\\vec{r}_0)|^3}

    Args:
        x, y, z: Position coordinates.
        x0, y0, z0: Monopole position
        nPhi0: Number of flux quanta contained in the monopole.

    Returns:
        The field at the given coordinates in units of ``Phi_0 / (length_units)**2``.
    """
    xp = x - x0
    yp = y - y0
    zp = z - z0
    Hz0 = zp / (xp ** 2 + yp ** 2 + zp ** 2) ** (3 / 2) / (2 * np.pi)
    return nPhi0 * Hz0


def MonopoleField(
    x0: float = 0, y0: float = 0, z0: float = 0, nPhi0: Union[int, float] = 1
) -> Parameter:
    """Returns a Parameter that computes the z-component of the field from a monopole
    (monopole) located at position ``(x0, y0, z0)`` containing a total of
    ``nPhi0`` flux quanta.

    .. math::

        \\mu_0H_z(\\vec{r}-\\vec{r}_0) = \\frac{n\\Phi_0}{2\\pi}
            \\frac{(\\vec{r}-\\vec{r}_0)\\cdot\\hat{z}}{|(\\vec{r}-\\vec{r}_0)|^3}

    Args:
        x0, y0, z0: Coordinates of the monopole position.
        nPhi0: Number of flux quanta contained in the monopole.

    Returns:
        A Parameter that returns the out-of-plane field in units of
        ``Phi_0 / (length_units)**2``.
    """
    return Parameter(monopole, x0=x0, y0=y0, z0=z0, nPhi0=nPhi0)


VortexField = MonopoleField


def pearl_vortex(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    z: Union[float, np.ndarray],
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    Lambda: float = 0,
    x0: float = 0,
    y0: float = 0,
    z0: float = 0,
    nPhi0: Union[int, float] = 1,
) -> Union[float, np.ndarray]:
    """The z-component of the field from a Pearl vortex.

    The field from a Pearl vortex located at is computed using a Fourier transform
    method. For a uniform thin film lying in the :math:`x-y` plane with effective
    penetration depth :math:`\\Lambda` (Pearl length :math:`2\\Lambda`), the Fourier
    transform of the :math:`z`-component of the field from a vortex containing
    :math:`n` flux quanta located at the origin, :math:`x=y=z=0`, is given by:

    .. math::

        \\mathcal{F}\\{\\mu_0H_z\\}(k_x, k_y, z) =
        \\frac{n\\Phi_0e^{-kz}}{1 + 2\\Lambda k},

    where :math:`k=\\sqrt{k_x^2 + k_y^2}` and the quantity is in units of
    ``Phi_0 / (length_units)**2``, where ``length_units`` are the units of
    ``xs``, ``ys``, etc. The field is calculated by inverse Fourier-transforming the
    above expression for an :math:`x-y` plane defined by parameters ``xs`` and ``ys``,
    then interpolating the field to the desired coordinates.

    .. seealso:: References: [Pearl-APL-1964]_, [Tafuri-PRL-2004]_.

    .. note::

        All elements of the array ``z`` must be equal. In other words, this function
        can only calculate the field :math:`\\mu_0H_z` evaluated at a plane parallel
        to the :math:`x-y` plane.

    Args:
        x, y, z: The coordinates at which to calculcate the field.
        x0, y0, z0: Coordinates of the Pearl vortex position.
        Lambda: The effective penetration depth of the film in which the vortex lies.
            ``Lambda`` is equal to half the Pearl length.
        nPhi0: Number of flux quanta contained in the monopole.
        xs, ys: Vectors of x and y coordinates defining the the domain in which the
            field will be computed using a Fourier transform as described above.

    Returns:
        The out-of-plane field :math:`\\mu_0H_z` evaluated at the given coordinates in
        units of ``Phi_0 / (length_units)**2``.
    """

    from scipy.interpolate import LinearNDInterpolator

    x, y, z = np.atleast_1d(x, y, z)
    if not np.allclose(z, z[0]):
        raise ValueError("All elements of the vector z must be equal.")
    x = x - x0
    y = y - y0
    z = np.abs(z[0] - z0)
    xs = np.sort(xs)
    ys = np.sort(ys)
    if (
        x.min() < xs.min()
        or x.max() > xs.max()
        or y.min() < ys.min()
        or y.max() > ys.max()
    ):
        raise ValueError(
            "The rectangle defined by xs and ys must contain the convex hull of the "
            "region defined by (x - x0) and (y - y0)."
        )
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    # Define Fourier-space coordinates
    kmaxx = np.pi / dx
    kmaxy = np.pi / dy
    kx = np.linspace(-kmaxx, kmaxx, xs.shape[0])
    ky = np.linspace(-kmaxy, kmaxy, ys.shape[0])
    X, Y = np.meshgrid(xs, ys)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)
    Lambda_Pearl = 2 * Lambda
    # Fourier transform of mu_0 * H_z(x, y, z)
    hzk = nPhi0 * np.exp(-K * z) / (1 + K * Lambda_Pearl)
    hzk = np.fft.fftshift(hzk)
    hz = np.abs(np.fft.fftshift(np.fft.ifft2(hzk))) / (dx * dy)
    # Interpolate to x, y, z coordinates
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    interp = LinearNDInterpolator(XY, hz.ravel())
    return interp(np.stack([x, y], axis=1)).squeeze()


def PearlVortexField(
    *,
    x0: float = 0,
    y0: float = 0,
    z0: float = 0,
    Lambda: float = 0,
    nPhi0: Union[int, float] = 1,
    xs: np.ndarray,
    ys: np.ndarray,
) -> Parameter:
    """Returns a Parameter that computes the z-component of the field from a Pearl
    vortex located at position ``(x0, y0, z0)`` in a film with effective penetration
    depth ``Lambda`` (Pearl length ``2 * Lambda``) containing a total of ``nPhi0``
    flux quanta.

    The field from a Pearl vortex located at is computed using a Fourier transform
    method. For a uniform thin film lying in the :math:`x-y` plane with effective
    penetration depth :math:`\\Lambda` (Pearl length :math:`2\\Lambda`), the Fourier
    transform of the :math:`z`-component of the field from a vortex containing
    :math:`n` flux quanta located at the origin, :math:`x=y=z=0`, is given by:

    .. math::

        \\mathcal{F}\\{\\mu_0H_z\\}(k_x, k_y, z) =
        \\frac{n\\Phi_0e^{-kz}}{1 + 2\\Lambda k},

    where :math:`k=\\sqrt{k_x^2 + k_y^2}` and the quantity is in units of
    ``Phi_0 / (length_units)**2``, where ``length_units`` are the units of
    ``xs``, ``ys``, etc. The field is calculated by inverse Fourier-transforming the
    above expression for an :math:`x-y` plane defined by parameters ``xs`` and ``ys``,
    then interpolating the field to the desired coordinates. Note that the
    Fourier method may not be accurate if ``xs`` and ``ys`` are not sampled finely
    enough.

    .. seealso:: References: [Pearl-APL-1964]_, [Tafuri-PRL-2004]_.

    Args:
        x0, y0, z0: Coordinates of the Pearl vortex position.
        Lambda: The effective penetration depth of the film in which the vortex lies.
            ``Lambda`` is equal to half the Pearl length.
        nPhi0: Number of flux quanta contained in the monopole.
        xs, ys: Vectors of x and y coordinates defining the the domain in which the
            field will be computed using a Fourier transform as described above.

    Returns:
        A Parameter that returns the out-of-plane field in units of
        ``Phi_0 / (length_units)**2``.
    """
    return Parameter(
        pearl_vortex,
        xs=xs,
        ys=ys,
        Lambda=Lambda,
        x0=x0,
        y0=y0,
        z0=z0,
        nPhi0=nPhi0,
    )
