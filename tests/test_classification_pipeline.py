from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from cassa.classification_pipeline import (
    eigen_decomposition,
    get_affinity_matrix,
    get_clusters_spectral,
)
from cassa.distance_matrix import compute_distance_matrix, compute_distance_matrix_dask

path = Path(__file__)


class TestDistMatrix(TestCase):
    def setUp(self):
        pass

    @pytest.mark.unit
    def test_distance_matrix(self):
        matrix_arrays = np.random.random((100, 10, 50))
        dist_matr = compute_distance_matrix(matrix_arrays)
        self.assertEqual(matrix_arrays.shape[0], dist_matr.shape[0])

        dist_matr_dask = compute_distance_matrix_dask(matrix_arrays, num_part=10)
        self.assertTrue((dist_matr == dist_matr_dask).all())

        aff_matrix = get_affinity_matrix(dist_matr)
        self.assertEqual(aff_matrix.shape, dist_matr.shape)

        n_cl = eigen_decomposition(aff_matrix)[0]
        self.assertGreater(n_cl, 0)

        l_labels, cl_colors, clusterer = get_clusters_spectral(
            dist_matr, ncl=n_cl, self_tuned=True
        )
        self.assertEqual(len(l_labels), len(cl_colors))

    def tearDown(self):
        pass
