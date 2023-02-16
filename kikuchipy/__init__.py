# Copyright 2019-2023 The kikuchipy developers
#
# This file is part of kikuchipy.
#
# kikuchipy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# kikuchipy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with kikuchipy. If not, see <http://www.gnu.org/licenses/>.

from typing import Union

from kikuchipy.release import version as __version__


# Attempt (and fail) import of optional dependencies only once
try:
    import pyvista

    _pyvista_installed = True
except ImportError:  # pragma: no cover
    _pyvista_installed = False

try:
    import nlopt

    _nlopt_installed = True
except ImportError:  # pragma: no cover
    _nlopt_installed = False

try:
    from pyebsdindex import pcopt, ebsd_index

    _pyebsdindex_installed = True
except ImportError:  # pragma: no cover
    _pyebsdindex_installed = False

# PyOpenCL compatible device available for use with PyEBSDIndex?
# PyOpenCL is an optional dependency of PyEBSDIndex, so it should not be
# an (optional) dependency of kikuchipy.
try:
    import pyopencl as cl

    platforms = cl.get_platforms()
    assert len(platforms) > 0
    _pyopencl_driver_available = True
except:  # pragma: no cover
    # Have to use bare except because PyOpenCL might raise its own
    # LogicError, but we also want to catch import errors here
    _pyopencl_driver_available = False


def set_log_level(level: Union[int, str]):  # pragma: no cover
    """Set level of kikuchipy logging messages.

    Parameters
    ----------
    level
        Any value accepted by :meth:`logging.Logger.setLevel()`. Levels
        are ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"`` and
        ``"CRITICAL"``.

    Notes
    -----
    See https://docs.python.org/3/howto/logging.html.

    Examples
    --------
    Note that you might have to set the logging level of the root stream
    handler to display kikuchipy's debug messages, as this handler might
    have been initialized by another package

    >>> import logging
    >>> logging.root.handlers[0]  # doctest: +SKIP
    <StreamHandler <stderr> (INFO)>
    >>> logging.root.handlers[0].setLevel("DEBUG")

    >>> import kikuchipy as kp
    >>> kp.set_log_level("DEBUG")
    >>> s = kp.data.nickel_ebsd_master_pattern_small()
    >>> s.set_signal_type("EBSD")  # doctest: +SKIP
    DEBUG:kikuchipy.signals._kikuchi_master_pattern:Delete custom attributes when setting signal type
    """
    import logging

    logging.basicConfig()
    logging.getLogger("kikuchipy").setLevel(level)


__all__ = [
    "__version__",
    "_pyebsdindex_installed",
    "_pyvista_installed",
    "data",
    "detectors",
    "draw",
    "filters",
    "generators",
    "imaging",
    "indexing",
    "io",
    "load",
    "pattern",
    "projections",
    "release",
    "set_log_level",
    "signals",
    "simulations",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    _import_mapping = {
        "load": "io._io",
    }
    if name in __all__:
        import importlib

        if name in _import_mapping.keys():
            import_path = f"{__name__}.{_import_mapping.get(name)}"
            return getattr(importlib.import_module(import_path), name)
        else:  # pragma: no cover
            return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
