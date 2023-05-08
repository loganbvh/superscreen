from dataclasses import dataclass
from typing import List

import h5py
import numpy as np
from scipy import integrate

from ..geometry import path_vectors
from .polygon import Polygon


@dataclass
class TerminalSet:
    """A container for the transport terminals for a single film.

    Args:
        film: The film name
        source_terminals: A list of :class:`superscreen.Polygon` instances
            representing the source terminals.
        drain_terminal: A :class:`superscreen.Polygon` representing the drain terminal.
    """

    film: str
    source_terminals: List[Polygon]
    drain_terminal: Polygon

    def to_list(self) -> List[Polygon]:
        """Return the terminals as a list. The drain terminal is the last element."""
        return list(self.source_terminals) + [self.drain_terminal]

    def copy(self) -> "TerminalSet":
        return TerminalSet(
            film=self.film,
            source_terminals=[poly.copy() for poly in self.source_terminals],
            drain_terminal=self.drain_terminal.copy(),
        )

    def to_hdf5(self, h5group: h5py.Group) -> None:
        """Save the TerminalSet to an h5py Group."""
        h5group.attrs["film"] = self.film
        for i, terminal in enumerate(self.to_list()):
            terminal.to_hdf5(h5group.create_group(str(i)))

    @staticmethod
    def from_hdf5(h5group: h5py.Group) -> "TerminalSet":
        """Load a TerminalSet from an h5py Group."""
        film = h5group.attrs["film"]
        terminals = []
        for i in range(len(h5group)):
            terminals.append(Polygon.from_hdf5(h5group[str(i)]))
        drain_terminal = terminals.pop(-1)
        return TerminalSet(
            film=film,
            source_terminals=terminals,
            drain_terminal=drain_terminal,
        )

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if not isinstance(other, TerminalSet):
            return False
        if other.film != self.film:
            return False
        for self_polygon, other_polygon in zip(self.to_list(), other.to_list()):
            if self_polygon != other_polygon:
                return False
        return True


def stream_from_current_density(points: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Computes the scalar stream function corresonding to a
    given current density :math:`J`, according to:

    .. math::

        g(\\vec{r})=g(\\vec{r}_0)+\\int_{\\vec{r}_0}^\\vec{r}
        (\\hat{z}\\times\\vec{J})\\cdot\\mathrm{d}\\vec{\\ell}

    Args:
        points: Shape ``(n, 2)`` array of ``(x, y)`` positions at which to
            compute the stream function :math:`g`.
        J: Shape ``(n, 2)`` array of the current density ``(Jx, Jy)`` at the
            given ``points``.s

    Returns:
        A shape ``(n, )`` array of the stream function at the given ``points``.
    """
    # (0, 0, 1) X (Jx, Jy, 0) == (-Jy, Jx, 0)
    zhat_cross_J = J[:, [1, 0]]
    zhat_cross_J[:, 0] *= -1
    dl = np.diff(points, axis=0, prepend=points[:1])
    integrand = np.sum(zhat_cross_J * dl, axis=1)
    return integrate.cumulative_trapezoid(integrand, initial=0)


def stream_from_terminal_current(points: np.ndarray, current: float) -> np.ndarray:
    """Computes the terminal stream function corresponding to a given terminal current.

    We assume that the current :math:`I` is uniformly distributed along the terminal
    with a current density :math:`\\vec{J}` which is perpendicular to the terminal.
    Then for :math:`\\vec{r}` along the terminal, the stream function is given by

    .. math::

        g(\\vec{r})=g(\\vec{r}_0)+\\int_{\\vec{r}_0}^\\vec{r}
        (\\hat{z}\\times\\vec{J})\\cdot\\mathrm{d}\\vec{\\ell}

    Args:
        points: A shape ``(n, 2)`` array of terminal vertex positions.
        current: The total current sources by the terminal.

    Returns:
        A shape ``(n, )`` array of the stream function along the terminal.
    """
    edge_lengths, unit_normals = path_vectors(points)
    J = current * unit_normals / np.sum(edge_lengths)
    g = stream_from_current_density(points, J)
    return g * current / g[-1]
