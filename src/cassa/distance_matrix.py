import logging

import dask.array as da
import numpy as np
from dask import compute, delayed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger("cassa-distance-matrix")


def compute_earth_mover_dist(first, second):
    """
    Compute earth's mover distance (EMD) between two data tensors.

    Parameters
    ----------
    first : np.ndarray
        First data array
    second : np.ndarray
        Second data array

    Returns
    ----------
    emd_val : float
        EMD distance between the two arrays
    """
    d = cdist(first, second)
    assignment = linear_sum_assignment(d)
    emd_val = d[assignment].sum()
    return emd_val


def compute_distance_matrix(matrix_arrays):
    """Compute distance matrix.

    Parameters
    ----------
    matrix_arrays : np.ndarray
        Matrix of data tensors stored in arrays.
        Only 1-D or 2-D data tensors allowed

    """
    # Get indices for the upper-triangle of matrix array
    indx, indy = np.triu_indices(len(matrix_arrays))
    np_arr = np.zeros((len(matrix_arrays), len(matrix_arrays)))

    if len(matrix_arrays.shape) == 2:
        # for a matrix of 1-D data tensors
        arr_1 = matrix_arrays[indx][:, np.newaxis]
        arr_2 = matrix_arrays[indy][:, np.newaxis]

    elif len(matrix_arrays.shape) == 3:
        # for a matrix of 2-D data tensors
        arr_1 = matrix_arrays[indx]
        arr_2 = matrix_arrays[indy]

    else:
        logger.error(" Distance matrix can be compute on 1-D and 2-D data tensors only")
        raise ValueError

    results = []
    for first, second in zip(arr_1, arr_2):
        res = compute_earth_mover_dist(first, second)
        results.append(res)

    np_arr[indx, indy] = np.array(results)
    # Construct lower-triangle (it is a symmetric matrix)
    i_lower = np.tril_indices(len(matrix_arrays), -1)
    np_arr[i_lower] = np_arr.T[i_lower]
    logger.info(" Constructed entire distance matrix")

    return np_arr


def _compute_earth_mover_distance(array_1, array_2):
    results = []
    for first, second in zip(array_1, array_2):
        res = compute_earth_mover_dist(first, second)
        results.append(res)
    return np.asarray(results)


def compute_distance_matrix_chunked(matrix_arrays, num_chunks=5, num_chunks_dask=500):
    """Compute distance matrix.

    Parameters
    ----------
    matrix_arrays : np.ndarray
        Matrix of data tensors stored in arrays.
        Only 1-D or 2-D data tensors allowed

    """
    # Get indices for the upper-triangle of matrix array
    indx, indy = np.triu_indices(len(matrix_arrays))
    np_arr = np.zeros((len(matrix_arrays), len(matrix_arrays)))

    indexes_chunks = np.linspace(0, len(indx), num_chunks).astype(int)

    arr_1_chunks = [
        da.from_array(
            matrix_arrays[indx][i:j],
            chunks=(num_chunks_dask, matrix_arrays.shape[1], matrix_arrays.shape[2]),
        )
        for i, j in zip(indexes_chunks[:-1], indexes_chunks[1:])
    ]
    arr_2_chunks = [
        da.from_array(
            matrix_arrays[indy][i:j],
            chunks=(num_chunks_dask, matrix_arrays.shape[1], matrix_arrays.shape[2]),
        )
        for i, j in zip(indexes_chunks[:-1], indexes_chunks[1:])
    ]

    results = []
    for a, b in zip(arr_1_chunks, arr_2_chunks):
        results.append(delayed(_compute_earth_mover_distance)(a, b))

    results = np.concatenate(compute(*results))

    np_arr[indx, indy] = results
    # Construct lower-triangle (it is a symmetric matrix)
    i_lower = np.tril_indices(len(matrix_arrays), -1)
    np_arr[i_lower] = np_arr.T[i_lower]
    logger.info(" Constructed entire distance matrix")

    return np_arr
