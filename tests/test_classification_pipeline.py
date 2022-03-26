from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from cassa.classification_pipeline import get_affinity_matrix
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

    def tearDown(self):
        pass
