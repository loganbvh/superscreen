import numpy as np


def auto_range_iqr(data_array, cutoff_percentile=90):
    """Get the min and max range of the provided array that excludes outliers
    following the IQR rule.

    This function computes the inter-quartile-range (IQR), defined by Q3-Q1,
    i.e. the percentiles for 75% and 25% of the distribution. The region
    without outliers is defined by [Q1-1.5*IQR, Q3+1.5*IQR].
    https://github.com/QCoDeS/Qcodes/blob/master/qcodes/utils/plotting.py.

    Args:
        data_array (np.ndarray): Array of arbitrary dimension containing the statistical data.
        cutoff_percentile (float | tuple[float]): Percentile of data that may maximally be clipped
            on both sides of the distribution. If given a tuple (a,b) the percentile limits will be a and 100-b.

    Returns:
        (tuple[float]): (vmin, vmax)
    """
    if isinstance(cutoff_percentile, tuple):
        t = cutoff_percentile[0]
        b = cutoff_percentile[1]
    else:
        t = cutoff_percentile
        b = cutoff_percentile
    z = data_array.flatten()
    zmax = np.nanmax(z)
    zmin = np.nanmin(z)
    zrange = zmax - zmin
    pmin, q3, q1, pmax = np.nanpercentile(z, [b, 75, 25, 100 - t])
    IQR = q3 - q1
    # handle corner case of all data zero, such that IQR is zero
    # to counter numerical artifacts do not test IQR == 0, but IQR on its
    # natural scale (zrange) to be smaller than some very small number.
    # also test for zrange to be 0.0 to avoid division by 0.
    # all This is possibly to careful...
    if zrange == 0.0 or IQR / zrange < 1e-8:
        vmin = zmin
        vmax = zmax
    else:
        vmin = max(q1 - 1.5 * IQR, zmin)
        vmax = min(q3 + 1.5 * IQR, zmax)
        # do not clip more than cutoff_percentile:
        vmin = min(vmin, pmin)
        vmax = max(vmax, pmax)
    return vmin, vmax
