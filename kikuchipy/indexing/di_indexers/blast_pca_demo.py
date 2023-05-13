import numpy as np
import pyopencl as cl
from pyopencl.array import Array
import pyclblast
from scipy.linalg.lapack import dsyevd
from typing import Union


def blast_covariance_matrix(data, dtype='float32'):
    """Compute the covariance matrix of A.

    Parameters
    ----------
    data : numpy.ndarray
        The array to compute the covariance matrix of.

    Returns
    -------
    numpy.ndarray
        The covariance matrix of A.

    """
    # Set up OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    # Set up NumPy arrays
    n_samples, n_features = data.shape
    data = data.astype(dtype)
    # Set up OpenCL array
    cl_a = Array(queue, data.shape, data.dtype)
    cl_a.set(data)
    # Prepare an empty OpenCL array for the result
    cl_cov = Array(queue, (n_features, n_features), dtype=dtype)
    # Perform the dsyrk operation
    pyclblast.syrk(queue, n_features, n_samples, cl_a, cl_cov, n_features, n_features, alpha=1.0, beta=0.0,
                   lower_triangle=False, a_transp=True)
    # Transfer result from device to host
    covariance_matrix = cl_cov.get()
    return covariance_matrix


def blast_transform(data, transform, dtype='float32'):
    """ Transform A using T.

    Parameters
    ----------
    data : numpy.ndarray
        The array to transform.
    transform : numpy.ndarray
        The transformation matrix.
    dtype : str or numpy.dtype, optional
        The dtype of the arrays. The default is 'float32'.

    Returns
    -------
    numpy.ndarray
        The transformed array.

    """
    # Set up OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    # Set up NumPy arrays
    n_samples, n_features = data.shape
    _, n_components = transform.shape
    data = data.astype(dtype)
    transform = transform.astype(dtype)
    # Set up OpenCL arrays
    cl_a = Array(queue, data.shape, data.dtype)
    cl_a.set(data)
    cl_t = Array(queue, transform.shape, transform.dtype)
    cl_t.set(transform)
    # Prepare an empty OpenCL array for the result
    cl_out = Array(queue, (n_samples, n_components), dtype=dtype)
    # Perform the gemm operation
    pyclblast.gemm(queue, n_samples, n_components, n_features, cl_a, cl_t, cl_out,
                   a_ld=n_features,
                   b_ld=n_components,
                   c_ld=n_components)
    queue.finish()
    output = cl_out.get()
    return output


class BlastPCA:
    """PCA using pyclblast to compute the covariance matrix.

    """

    def __init__(self, n_components: int, whiten: bool = True, datatype: Union[str, np.dtype] = 'float32'):
        """Initialize the BlastPCA object.

        Parameters
        ----------
        n_components : int, optional
            The number of components to keep. The default is None.
        datatype : str or numpy.dtype, optional
            The dtype of the data. The default is 'float32'.
        """
        self.n_components = n_components
        self.whiten = whiten
        # check the datatype
        if isinstance(datatype, str):
            assert datatype in ['float64', 'float32']
        elif isinstance(datatype, np.dtype):
            assert datatype in [np.float64, np.float32]
        else:
            raise TypeError("dtype must be a string or a numpy dtype but got %s" % type(datatype))
        self.datatype = datatype
        self._eigenvectors = None
        self._components = None
        self._eigenvalues = None
        self._singular_values_ = None
        self._explained_variance_ = None
        self._explained_variance_ratio_ = None
        self._transformation_matrix = None
        self._inverse_transformation_matrix = None

    def fit(self, data):
        """Fit the PCA to the data.

        Parameters
        ----------
        data : numpy.ndarray
            The data to fit the PCA to.

        """
        # check the data dimensons
        if data.ndim != 2:
            raise ValueError("A must be a 2D array but got %d dimensions" % data.ndim)
        n_samples, n_features = data.shape
        # check the n_components is less than the number of features
        if self.n_components > n_features:
            raise ValueError(f"n_components, {self.n_components} given, exceeds features in A, {n_features}")
        # compute the covariance matrix
        cov = blast_covariance_matrix(data, dtype=self.datatype)
        # compute the eigenvalues and eigenvectors
        self._eigenvalues, self._eigenvectors, info = dsyevd(cov)
        # if the info is not 0, then something went wrong
        if info != 0:
            raise RuntimeError("dsyevd (symmetric eigenvalue decomposition call to LAPACK) failed with info=%d" % info)
        # sort the eigenvalues and eigenvectors
        idx = np.argsort(self._eigenvalues)[::-1]
        self._eigenvalues = self._eigenvalues[idx]
        self._eigenvectors = self._eigenvectors[:, idx]
        self._singular_values_ = np.sqrt(self._eigenvalues)[:self.n_components]
        # keep only the first n_components and make sure the arrays are contiguous in C order
        self._eigenvalues = np.ascontiguousarray(self._eigenvalues[:self.n_components])
        self._eigenvectors = np.ascontiguousarray(self._eigenvectors[:, :self.n_components])
        # calculate the explained variance
        self._explained_variance_ = self._eigenvalues / (n_samples - 1)
        self._explained_variance_ratio_ = self._explained_variance_ / np.sum(self._explained_variance_)
        # components is an alias for the un-whitened eigenvectors
        # calculate the transformation matrix
        if self.whiten:
            self._transformation_matrix = self._eigenvectors * np.sqrt(n_samples-1) / self._singular_values_[np.newaxis, :]
            self._inverse_transformation_matrix = np.linalg.pinv(self._transformation_matrix)
        else:
            self._transformation_matrix = self._eigenvectors
            self._inverse_transformation_matrix = np.linalg.pinv(self._transformation_matrix)

    def fit_transform(self, data):
        """Fit the PCA to the data and transform the data.

        Parameters
        ----------
        data : numpy.ndarray
            The data to fit the PCA to.

        Returns
        -------
        numpy.ndarray
            The transformed data.

        """
        self.fit(data)
        return self.transform(data)

    def transform(self, data: np.array, mean_center: bool = False):
        """Apply the dimensionality reduction on data.

        Parameters
        ----------
        data : numpy.ndarray
            The input data.
        mean_center : bool, optional
            Whether to mean center the data. The default is False.

        Returns
        -------
        numpy.ndarray
            The transformed data.

        """
        # Mean centering
        if mean_center:
            data = data - np.mean(data, axis=0)
        # Project the data onto the principal components
        transformed_data = blast_transform(data, self._transformation_matrix, dtype=self.datatype)
        return transformed_data

    def inverse_transform(self, data: np.array):
        """Transform data back to its original space.

        Parameters
        ----------
        data : numpy.ndarray
            The transformed data.

        Returns
        -------
        numpy.ndarray
            The original data.

        """
        return blast_transform(data, self._inverse_transformation_matrix, dtype=self.datatype)

    @property
    def components_(self):
        """Return the principal components (eigenvectors).

        """
        return self._eigenvectors.T

    @property
    def explained_variance_(self):
        """Return the explained variance (eigenvalues).

        """
        return self._explained_variance_

    @property
    def singular_values_(self):
        """Return the singular values.

        """
        return self._singular_values_


# show that inverse transform is the same with and without whitening
def look_inverse_transform():
    # create a random dataset
    n_samples = 100
    n_features = 5
    n_components = 3
    # set numpy seed
    np.random.seed(0)
    data = np.random.rand(n_samples, n_features).astype('float32')
    data = data - np.mean(data, axis=0)

    # create the PCA object with whitening
    pca = BlastPCA(n_components=n_components, whiten=True)
    # fit the PCA
    pca.fit(data)
    # transform the data
    transformed_data_whiten = pca.transform(data)
    # inverse transform the data
    inverse_transformed_data_whiten = pca.inverse_transform(transformed_data_whiten)

    # create the PCA object without whitening
    pca = BlastPCA(n_components=n_components, whiten=False)
    # fit the PCA
    pca.fit(data)
    # transform the data
    transformed_data_no_whiten = pca.transform(data)
    # inverse transform the data
    inverse_transformed_data_no_whiten = pca.inverse_transform(transformed_data_no_whiten)

    print("Inverse transform with whitening")
    print(inverse_transformed_data_whiten[0])
    print("Inverse transform without whitening")
    print(inverse_transformed_data_no_whiten[0])


    # check that the data is the same
    assert np.allclose(inverse_transformed_data_no_whiten, inverse_transformed_data_whiten, atol=1e-1)

look_inverse_transform()