import logging
import sys

import numpy as np
import seaborn as sns
from numpy import linalg as LA
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering

logger = logging.getLogger("cassa-classification-pipeline")


def get_affinity_matrix(distance_matrix, k=7):
    """
    Compute affinity matrix based on distance matrix and
    apply local scaling based on the k nearest neighbour.

    Parameters
    ----------

    distance_matrix : np.ndarray
        Distance matrix (symmetric matrix)
    k : int
        k-th position for local statistics of the neighborhood
        surrounding each point (local scaling parameter for each
        point allows self-tuning of the point-to-point distances)

    Returns
    ----------
    affinity_matrix : np.ndarray
        affinity_matrix

    References
    ----------
    .. [1] https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    .. [2] code adapted from:
           https://github.com/ciortanmadalina/high_noise_clustering/blob/master/spectral_clustering.ipynb
    """
    # distance matrix (n x n)
    dists = distance_matrix
    # for each row, sort the distances ascendingly and take the index of the
    # k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    # create column vector (n x 1)
    knn_distances = knn_distances[np.newaxis].T
    # create a simmeric matrix (n x n) with rank=1
    # by column times row vectors moltiplication
    local_scale = knn_distances.dot(knn_distances.T)
    # divide square distance matrix by local scale
    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


def eigen_decomposition(A, topK=5):
    """
    Compute optimum number of clusters for Spectral Clustering based on eigengap heuristic algorithm.

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters by eigengap heuristic

    Parameters
    ----------

    A : np.ndarray
        affinity matrix
    topK : int
        top optimal number of clusters to return

    Returns
    ----------
    nb_clusters : list[int]
        topK optimal numbers of clusters by eigengap heuristic


    References
    ----------
    .. [1] https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    .. [2] http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh),
    # that is, largest eigenvalues in  the euclidean norm of complex numbers.
    eigenvalues, eigenvectors = LA.eig(L)

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigenvalues
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters


def get_clusters_spectral(distance_matrix, ncl=0, dlt=0, self_tuned=False):
    """
    Compute clusters with Spectral Clustering on a precomputed distance matrix.

    Parameters
    ----------

    distance_matrix : np.ndarray
        distance matrix
    ncl : int
        number of clusters
    dlt : float
        delta parameter
    k : int
        k-th position for local statistics of the neighborhood
    self_tuned : bool
        activate self-tuning

    Returns
    ----------
     tuple : tuple[np.ndarray, list, sklearn.cluster._spectral.SpectralClustering]
        - cl_labels is an array of clusters labels
        - cl_colors is an list of clusters colors
        - clusterer is the clusterer
    """
    if self_tuned:
        affinity_matrix = get_affinity_matrix(distance_matrix)
        if ncl == 0:
            k = eigen_decomposition(affinity_matrix)
            ncl = k[0]
            logger.info(f" Optimal n. of clusters {ncl}")
            logger.info("*" * 40)
        clusterer = SpectralClustering(n_clusters=ncl, affinity="precomputed").fit(
            affinity_matrix
        )

    else:
        if ncl == 0:
            logger.error("*" * 40)
            logger.error(" ncl cannot be zero unless self_tuned is True.")
            logger.error("*" * 40)
            sys.exit(1)
        if dlt == 0:
            logger.error("*" * 40)
            logger.error(" delta cannot be zero unless self_tuned is True.")
            logger.error("*" * 40)
            sys.exit(1)
        affinity_matrix = np.exp(-(distance_matrix ** 2) / (2.0 * dlt ** 2))
        clusterer = SpectralClustering(n_clusters=ncl, affinity="precomputed").fit(
            affinity_matrix
        )

    cl_labels = clusterer.labels_
    palette = sns.color_palette("deep", np.unique(cl_labels).max() + 1)
    cl_colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in cl_labels]
    return cl_labels, cl_colors, clusterer


def get_clusters_hdbscan(par_1, par_2, par_3, par_4, distance_matrix):
    """
    Compute clusters with HDBSCAN on a precomputed distance matrix between
    wavelet spectra obtained from RO profiles.

    Parameters
    ----------
    par_1 : int
        min_cluster_size - The minimum size of clusters; single linkage splits that contain fewer points
        than this will be considered points "falling out" of a cluster rather than
        a cluster splitting into two new clusters.
    par_2 : int
        min_samples - The number of samples in a neighbourhood for a point to be considered a core point.
    par_3 : string
        The method used to select clusters from the condensed tree.
        The standard approach for HDBSCAN is to use an Excess of Mass algorithm to find the most persistent clusters.
        Alternatively you can instead select the clusters at the leaves of the tree – this provides
        the most fine grained and homogeneous clusters. Options are: eom and leaf. Default is eom
    par_4 : float
        cluster_selection_method - A distance threshold. Clusters below this value will be merged.
    distance_matrix : np.ndarray
        Precomputed distance matrix

    Returns
    ----------
     tuple : tuple[np.ndarray, list, sklearn.cluster._spectral.SpectralClustering]
        - cl_labels is an array of clusters labels
        - cl_colors is an list of clusters colors
        - clusterer is the clusterer
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=par_1,
        min_samples=par_2,
        cluster_selection_method=par_3,
        cluster_selection_epsilon=par_4,
    )

    clusterer.fit(distance_matrix)

    cl_labels = clusterer.labels_
    palette = sns.color_palette("deep", np.unique(cl_labels).max() + 1)
    cl_colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in cl_labels]
    logging.info(f"N. of clusters with HDBSCAN: { len(np.unique(cl_labels)) }")
    return cl_labels, cl_colors, clusterer
