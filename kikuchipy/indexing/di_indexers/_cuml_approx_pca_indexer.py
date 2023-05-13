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

"""Private tools for custom dictionary indexing of experimental
 patterns to a dictionary of simulated patterns with known orientations.
"""

from typing import List, Union, Optional
import numpy as np

# try to import CuHNSW and notify user if not available
try:
    from cuhnsw import CuHNSW
# except all exceptions to avoid import errors
except ImportError:
    CuHNSW = None
    print(
        "cuhnsw is not installed. "
        "CUHNSW will not be available for use."
    )

# try to import cuml and notify user if not available
try:
    from cuml import PCA
except ImportError:
    cuml = None
    print(
        "cuml is not installed. "
        "Will not be available for use."
    )

from kikuchipy.indexing.di_indexers._di_indexer import DIIndexer


class CumlApproxPCAIndexer(DIIndexer):
    """Dictionary indexing using cuml.

    Summary
    -------
    This class implements a function to match experimental and simulated
    EBSD patterns in a dictionary using cuml. The dictionary must be able
    to fit in memory. The dictionary is indexed using the cuml library,
    specifically the NearestNeighbors class. More information can be found
    at the cuml documentation:

    https://docs.rapids.ai/api/cuml/stable/api.html#nearest-neighbors

    """

    _allowed_dtypes: List[type] = [np.float32, np.float64]

    def __init__(self,
                 datatype: Union[str, np.dtype, type],
                 normalize: bool = True,
                 zero_mean: bool = True,
                 space: str = "cosine",
                 n_components: int = 1000,
                 whiten: bool = True,
                 ef_construction: int = 300,
                 ef_search: int = 300,
                 max_m: int = 16,
                 max_m0: int = 32,
                 load_filename: Optional[str] = None,
                 save_filename: Optional[str] = None,
                 ):
        """Initialize the HNSWlib indexer.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type of the dictionary and experimental patterns.
        normalize : bool, optional
            Whether to normalize the patterns in the dictionary and
            experimental patterns. Default is True.
        zero_mean : bool, optional
            Whether to zero mean the patterns in the dictionary and
            experimental patterns. Default is True.
        space : str, optional
            The metric space to use for the nearest neighbors search.
            Default is "cosine".
        n_components : int, optional
            The number of components to use for PCA. Default is 1000.
        whiten : bool, optional
            Whether to whiten the PCA components. Default is True.
        ef_construction : int, optional
            The construction speed-accuracy tradeoff. Higher is slower
            but more accurate.
            Default is 300.
        ef_search : int, optional
            The query search speed-accuracy tradeoff. Higher is slower
            but more accurate.
            Default is 300.
        max_m : int, optional
            The maximum number of outgoing edges from a node during
            construction. Default is 16.
        max_m0 : int, optional
            The maximum number of outgoing edges from a node during
            construction of the zero layer. Default is 32.
        load_filename : str, optional
            The filename to load the index from. Default is None.
        save_filename : str, optional
            The filename to save the index to. Default is None.
        """
        super().__init__(datatype=datatype)
        self._normalize = normalize
        self._zero_mean = zero_mean
        self.space = space
        self.n_components = n_components
        self.whiten = whiten
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_m = max_m
        self.max_m0 = max_m0
        self.load_filename = load_filename
        self.save_filename = save_filename
        self._pca = None
        self._graph = None

    def prepare_dictionary(self, dictionary_patterns: np.ndarray):
        """Prepare the dictionary_patterns for indexing.

        """
        if self._normalize:
            self.dictionary_patterns = _normalize_patterns(dictionary_patterns)
        if self._zero_mean:
            self.dictionary_patterns = _zero_mean_patterns(dictionary_patterns)

        # change dictionary to datatype
        self.dictionary_patterns = self.dictionary_patterns.astype(self.datatype)

        # make incremental PCA object
        self._pca = PCA(n_components=self.n_components, whiten=self.whiten)

        # make the cuml nearest neighbors object
        dictionary_pca_components = self._pca.fit_transform(self.dictionary_patterns)

        # fit the knn lookup object
        self._graph = _build_index(
            dictionary_pca_components,
            space=self.space,
            ef_construction=self.ef_construction,
            max_m=self.max_m,
            max_m0=self.max_m0,
            load_filename=self.load_filename,
            save_filename=self.save_filename,
        )

    def prepare_experimental(self, experimental_patterns: np.ndarray) -> np.ndarray:
        """Prepare the experimental_patterns for indexing.

        """
        experimental_patterns = experimental_patterns.astype(self.datatype)
        if self._normalize:
            experimental_patterns = _normalize_patterns(experimental_patterns)
        if self._zero_mean:
            experimental_patterns = _zero_mean_patterns(experimental_patterns)
        return experimental_patterns

    def query(self, experimental_patterns: np.ndarray):
        """Query the dictionary.

        Parameters
        ----------
        experimental_patterns : numpy.ndarray
            Experimental patterns to match to the dictionary.

        Returns
        -------
        numpy.ndarray
            The indices of the dictionary patterns that match the experimental
            patterns.

        """
        experimental_patterns = self.prepare_experimental(experimental_patterns)
        experimental_patterns_pca_components = self._pca.transform(experimental_patterns)
        return _search_index(self._graph,
                             self.keep_n,
                             experimental_patterns_pca_components,
                             self.ef_search,
                             )


def _build_index(dictionary_patterns,
                 space: str,
                 ef_construction: int,
                 max_m: int,
                 max_m0: int,
                 load_filename: Optional[str] = None,
                 save_filename: Optional[str] = None
                 ):
    """Build the index using the HNSWlib library.

    Parameters
    ----------
    dictionary_patterns : numpy.ndarray
        Dictionary patterns to index.
    kwargs
        Additional keyword arguments to pass to the HNSWlib library.
        See the HNSWlib GitHub repository for more information.

    Returns
    -------
    hnswlib.Index
        The index built using the HNSWlib library.

    """
    hnsw_graph = CuHNSW(opt={"distType": space,
                             "maxM": max_m,
                             "maxM0": max_m0,
                             "efConstruction": ef_construction
                             })
    if load_filename is not None:
        hnsw_graph.load_index(load_filename)
    else:
        hnsw_graph.set_data(dictionary_patterns)
        hnsw_graph.build()
        if save_filename is not None:
            hnsw_graph.save_index(save_filename)
    return hnsw_graph


def _search_index(index: CuHNSW,
                  keep_n: int,
                  experimental_patterns: np.ndarray,
                  ef_search: int
                  ):
    """Search the index using the HNSWlib library.

    Parameters
    ----------
    index : hnswlib.Index
        The index built using the HNSWlib library.
    keep_n : int
        The number of dictionary patterns to return for each experimental
        pattern.
    experimental_patterns : numpy.ndarray
        Experimental patterns to search.

    Returns
    -------
    numpy.ndarray
        The indices of the dictionary patterns that match the
        experimental patterns.
    numpy.ndarray
        The distances of the dictionary patterns that match the
        experimental patterns.

    """
    indices, distances, _ = index.search_knn(experimental_patterns,
                                             topk=keep_n,
                                             ef_search=ef_search,
                                             )
    return indices, distances


def _normalize_patterns(patterns: np.ndarray) -> np.ndarray:
    """Normalize the patterns.
    """
    patterns /= np.linalg.norm(patterns, axis=1)[:, None]
    return patterns


def _zero_mean_patterns(patterns: np.ndarray) -> np.ndarray:
    """Zero mean the patterns.
    """
    patterns -= np.mean(patterns, axis=1)[:, None]
    return patterns
