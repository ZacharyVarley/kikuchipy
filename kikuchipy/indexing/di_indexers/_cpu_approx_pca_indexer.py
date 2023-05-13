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

from joblib import Parallel, delayed, cpu_count

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

import hnswlib

from kikuchipy.indexing.di_indexers._di_indexer import DIIndexer


class ApproxPCAIndexer(DIIndexer):
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
                 dtype: Union[str, np.dtype, type],
                 n_threads: int = -2,
                 normalize: bool = True,
                 zero_mean: bool = True,
                 space: str = "cosine",
                 n_components: int = 700,
                 whiten: bool = True,
                 ef_construction: int = 200,
                 ef_search: int = 200,
                 max_m: int = 16,
                 load_filename: Optional[str] = None,
                 save_filename: Optional[str] = None,
                 ):
        """Initialize the HNSWlib indexer.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type of the dictionary and experimental patterns.
        n_threads : int, optional
            Number of threads to use for the nearest neighbors search. The default is -2.
            -2 means all but one thread, -1 means all threads
        normalize : bool, optional
            Whether to normalize the patterns before indexing. The default is True.
        zero_mean : bool, optional
            Whether to zero mean the patterns before indexing. The default is True.
        space : str, optional
            The metric to use for the nearest neighbors search. The default is "cosine".
        n_components : int, optional
            The number of components to keep in the PCA. The default is 700.
        whiten : bool, optional
            Whether to whiten the PCA. The default is True.
        ef_construction : int, optional
            The construction speed-accuracy tradeoff. Higher is slower
            but more accurate.
            Default is 200.
        ef_search : int, optional
            The query search speed-accuracy tradeoff. Higher is slower
            but more accurate.
            Default is 200.
        max_m : int, optional
            The maximum number of outgoing edges from a node during
            construction. Default is 16.
        load_filename : str, optional
            The filename to load the index from. Default is None.
        save_filename : str, optional
            The filename to save the index to. Default is None.
        """
        super().__init__(dtype=dtype)
        self._normalize = normalize
        self._zero_mean = zero_mean
        self.space = space
        self.n_components = n_components
        self.whiten = whiten
        self.n_threads = n_threads if n_threads > 0 else (cpu_count() + n_threads + 1)
        self._pca = None
        self._graph = None
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_m = max_m
        self.load_filename = load_filename
        self.save_filename = save_filename

    def prepare_dictionary(self, dictionary_patterns: np.ndarray):
        """Prepare the dictionary_patterns for indexing.

        """
        if self._normalize:
            dictionary_patterns = _normalize_patterns(dictionary_patterns)
        if self._zero_mean:
            dictionary_patterns = _zero_mean_patterns(dictionary_patterns)

        # make incremental PCA object
        self._pca = PCA(n_components=self.n_components, whiten=self.whiten)

        # make the cuml nearest neighbors object
        dictionary_pca_components = self._pca.fit_transform(dictionary_patterns)

        # don't need dictionary patterns anymore
        del dictionary_patterns

        # fit the knn lookup object
        self._graph = hnswlib.Index(space=self.space,
                                    dim=dictionary_pca_components.shape[1])
        self._graph.init_index(max_elements=dictionary_pca_components.shape[0],
                               ef_construction=self.ef_construction,
                               M=self.max_m)
        self._graph.add_items(dictionary_pca_components)

        # don't need pca components anymore
        del dictionary_pca_components

    def prepare_experimental(self, experimental_patterns: np.ndarray) -> np.ndarray:
        """Prepare the experimental_patterns for indexing.

        """
        experimental_patterns = experimental_patterns.astype(self.dtype)
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
        # parallelize the pca transform using joblib if len(experimental_patterns) > 1000
        if len(experimental_patterns) > self.n_threads:
            split = np.array_split(experimental_patterns, self.n_threads)
            parallel = Parallel(n_jobs=self.n_threads)
            exp_pca = np.concatenate(parallel(delayed(self._pca.transform)(split[i]) for i in range(self.n_threads)))
        else:
            exp_pca = self._pca.transform(experimental_patterns)
        self._graph.set_ef(self.ef_search)
        indices, distances = self._graph.knn_query(exp_pca, k=self.keep_n)
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
