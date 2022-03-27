import logging

import numpy as np
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