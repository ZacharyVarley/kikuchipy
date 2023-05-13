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

from typing import List, Union
import numpy as np
import dask.array as da

# try to import cuml and notify user if not available
try:
    import cudf
    from cuml import IncrementalPCA, NearestNeighbors
except ImportError:
    cuml = None
    print(
        "cuml is not installed. "
        "Will not be available for use."
    )

from kikuchipy.indexing.di_indexers._di_indexer import DIIndexer


class CumlIPCAIndexer(DIIndexer):
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
                 normalize: bool = True,
                 zero_mean: bool = True,
                 space: str = "cosine",
                 n_components: int = 100,
                 batch_size: int = 10000,
                 whiten: bool = True,
                 ):
        """Initialize the HNSWlib indexer.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type of the dictionary and experimental patterns.
        kwargs
            Additional keyword arguments to pass to the HNSWlib library.
            See the HNSWlib GitHub repository for more information.

        """
        super().__init__(dtype=dtype)
        self._normalize = normalize
        self._zero_mean = zero_mean
        self.space = space
        self.n_components = n_components
        self.batch_size = batch_size
        self.whiten = whiten
        self._fitted_PCA = None
        self._dictionary_pca_components = None
        self._knn_lookup_object = None

    def prepare_dictionary(self):
        """Prepare the dictionary_patterns for indexing.

        """
        if self._normalize:
            self.dictionary_patterns = _normalize_patterns(self.dictionary_patterns)
        if self._zero_mean:
            self.dictionary_patterns = _zero_mean_patterns(self.dictionary_patterns)

        # make incremental PCA object
        pca = IncrementalPCA(n_components=self.n_components, batch_size=self.batch_size, whiten=self.whiten)

        # loop over the (presumably) UINT8 dictionary in batches to fit the PCA
        for i in range(0, self.dictionary_patterns.shape[0], self.batch_size):
            if i + self.batch_size < self.dictionary_patterns.shape[0]:
                batch = self.dictionary_patterns[i:i + self.batch_size]
            else:
                batch = self.dictionary_patterns[i:]
            # if batch is lazy dask array, compute it
            if isinstance(batch, da.Array):
                batch = batch.compute()
            batch = batch.astype(self.dtype)
            batch = cudf.DataFrame(batch)
            pca.partial_fit(batch)

        # make the cuml nearest neighbors object
        self._fitted_PCA = pca

        # again loop over the dictionary in batches to transform the patterns
        for i in range(0, self.dictionary_patterns.shape[0], self.batch_size):
            if i + self.batch_size < self.dictionary_patterns.shape[0]:
                batch = self.dictionary_patterns[i:i + self.batch_size]
            else:
                batch = self.dictionary_patterns[i:]
            # if batch is lazy dask array, compute it
            if isinstance(batch, da.Array):
                batch = batch.compute()
            batch = batch.astype(self.dtype)
            batch = cudf.DataFrame(batch)
            batch = self._fitted_PCA.transform(batch)
            if i == 0:
                self._dictionary_pca_components = batch
            else:
                self._dictionary_pca_components = np.vstack((self._dictionary_pca_components, batch))

        self._knn_lookup_object = NearestNeighbors(n_neighbors=self.keep_n, metric=self.space)
        self._knn_lookup_object.fit(self._dictionary_pca_components)

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
        experimental_pca_components = self._fitted_PCA.transform(experimental_patterns)
        distances, indices = self._knn_lookup_object.kneighbors(experimental_pca_components)
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
