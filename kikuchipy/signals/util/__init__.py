# Copyright 2019-2022 The kikuchipy developers
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

"""Signal utilities for handling signal metadata and attributes, output
from signal methods, and for controlling chunking of lazy signal data in
:class:`~dask.array.Array`.
"""

__all__ = [
    "ebsd_metadata",
    "get_chunking",
    "get_dask_array",
    "metadata_nodes",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    _import_mapping = {
        "ebsd_metadata": "_metadata",
        "get_chunking": "_dask",
        "get_dask_array": "_dask",
        "metadata_nodes": "_metadata",
    }
    if name in __all__:
        import importlib

        if name in _import_mapping.keys():
            import_path = f"{__name__}.{_import_mapping.get(name)}"
            return getattr(importlib.import_module(import_path), name)
        else:  # pragma: no cover
            return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
