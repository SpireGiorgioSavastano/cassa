from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from cassa.classification_pipeline import (
    eigen_decomposition,
    get_affinity_matrix,
    get_clusters_spectral,
)
from cassa.distance_matrix import (
    compute_distance_matrix,
    compute_distance_matrix_chunked,
)

path = Path(__file__)


class TestDistMatrix(TestCase):
    def setUp(self):
        pass

    @pytest.mark.unit
    def test_distance_matrix(self):
        matrix_arrays = np.random.random((100, 10, 50))
        dist_matr = compute_distance_matrix(matrix_arrays)
        self.assertEqual(matrix_arrays.shape[0], dist_matr.shape[0])

        aff_matrix = get_affinity_matrix(dist_matr)
        self.assertEqual(aff_matrix.shape, dist_matr.shape)

        n_cl = eigen_decomposition(aff_matrix)[0]
        self.assertGreater(n_cl, 0)

        l_labels, cl_colors, clusterer = get_clusters_spectral(
            dist_matr, ncl=n_cl, self_tuned=True
        )
        self.assertEqual(len(l_labels), len(cl_colors))

    @pytest.mark.unit
    def test_distance_matrix_chunked(self):
        matrix_arrays = np.random.random((100, 10, 50))
        dist_matr_1 = compute_distance_matrix(matrix_arrays)
        dist_matr_2 = compute_distance_matrix_chunked(matrix_arrays)

        self.assertTrue((dist_matr_1 == dist_matr_2).all())

    def tearDown(self):
        pass
