from itertools import permutations
from unittest import TestCase

import numpy as np
import numpy.testing as npt

from distancematrix.generator import ZNormEuclidean
from distancematrix.consumer import MatrixProfileLR
from distancematrix.calculator import AnytimeCalculator
from distancematrix.ostinato import find_consensus_motif, CMResult


class TestOstinato(TestCase):
    def test_exact_match(self):
        # Each series contains a shifted/scaled version of [1, 1, 0, 2, 2]
        series_list = np.array([
            np.array([0.04, 0.45, 0.45, 0.00, 0.90, 0.90, 0.74, 0.72, 0.48, 0.82, 0.49, 0.36, 0.02, 0.37, 0.21]),
            np.array([0.08, 0.19, 0.25, 0.59, 0.50, 0.72, 0.16, 0.45, 1.49, 1.49, 0.49, 2.49, 2.49, 0.92, 0.16]),
            np.array([0.29, 0.42, 0.96, 1.68, 1.68, 1.00, 2.36, 2.36, 0.14, 0.22, 0.51, 0.45, 0.01, 0.66, 0.53]),
            np.array([0.84, 0.01, 0.01, 0.00, 0.02, 0.02, 0.51, 0.53, 0.91, 0.94, 0.47, 0.36, 0.28, 0.15, 0.08])
        ])

        correct_subseq_idx = [1, 8, 3, 1]

        for perm in permutations(range(len(series_list))):
            perm = list(perm)  # Tuple to list for indexing
            calc_result = find_consensus_motif(series_list[perm], 5)
            bf_result = find_consensus_motif_bruteforce(series_list[perm], 5)

            npt.assert_almost_equal(bf_result.radius, 0)
            npt.assert_equal(bf_result.series_index, 0)
            npt.assert_equal(bf_result.subseq_index, correct_subseq_idx[perm[0]])

            npt.assert_almost_equal(calc_result.radius, 0)
            npt.assert_equal(calc_result.series_index, 0)
            npt.assert_equal(calc_result.subseq_index, correct_subseq_idx[perm[0]])

    def test_near_match(self):
        # Fourth series contains shifted/scaled [1, 1, 1, 2, 2],
        # all other series contain shifted/scaled versions with slight noise.
        series_list = np.array([
            np.array([0.04, 0.40, 0.50, 0.45, 0.90, 0.90, 0.74, 0.72, 0.48, 0.82, 0.49, 0.36, 0.02, 0.37, 0.21]),
            np.array([0.08, 0.19, 0.25, 0.59, 0.50, 0.72, 0.16, 0.45, 1.53, 1.44, 1.49, 2.49, 2.49, 0.92, 0.16]),
            np.array([0.29, 0.42, 0.96, 1.68, 1.78, 1.58, 2.36, 2.36, 0.14, 0.22, 0.51, 0.45, 0.01, 0.66, 0.53]),
            np.array([0.84, 0.01, 0.01, 0.01, 0.02, 0.02, 0.51, 0.53, 0.91, 0.94, 0.47, 0.36, 0.28, 0.15, 0.08])
        ])

        for perm in permutations(range(len(series_list))):
            perm = list(perm)  # Tuple to list for indexing
            calc_result = find_consensus_motif(series_list[perm], 5)
            bf_result = find_consensus_motif_bruteforce(series_list[perm], 5)

            npt.assert_almost_equal(calc_result.radius, bf_result.radius)
            npt.assert_equal(bf_result.series_index, perm.index(3))
            npt.assert_equal(calc_result.series_index, perm.index(3))
            npt.assert_equal(bf_result.subseq_index, 1)
            npt.assert_equal(calc_result.subseq_index, 1)

    def test_on_random_data(self):
        data = np.array([
            [0.292, 0.183, 0.509, 0.128, 0.718, 0.054, 0.7, 0.532, 0.178, 0.076, 0.46, 0.027, 0.882, 0.288, 0.746],
            [0.57, 0.539, 0.239, 0.328, 0.784, 0.614, 0.288, 0.696, 0.12, 0.337, 0.54, 0.401, 0.589, 0.461, 0.666],
            [0.454, 0.487, 0.687, 0.981, 0.24, 0.863, 0.458, 0.203, 0.798, 0.917, 0.336, 0.562, 0.266, 0.325, 0.818],
            [0.749, 0.886, 0.095, 0.335, 0.247, 0.403, 0.063, 0.047, 0.804, 0.976, 0.836, 0.065, 0.27, 0.59, 0.747],
            [0.196, 0.924, 0.968, 0.19, 0.999, 0.31, 0.908, 0.576, 0.521, 0.246, 0.444, 0.319, 0.781, 0.628, 0.183],
            [0.136, 0.444, 0.115, 0.954, 0.231, 0.876, 0.566, 0.886, 0.898, 0.287, 0.544, 0.365, 0.108, 0.345, 0.03],
            [0.813, 0.324, 0.465, 0.459, 0.565, 0.28, 0.334, 0.169, 0.479, 0.957, 0.621, 0.026, 0.998, 0.732, 0.365],
            [0.176, 0.072, 0.288, 0.915, 0.867, 0.215, 0.566, 0.555, 0.602, 0.943, 0.786, 0.404, 0.271, 0.579, 0.362],
            [0.7, 0.113, 0.159, 0.701, 0.476, 0.216, 0.359, 0.613, 0.358, 0.871, 0.888, 0.668, 0.604, 0.574, 0.555],
            [0.745, 0.298, 0.213, 0.669, 0.303, 0.737, 0.93, 0.998, 0.529, 0.215, 0.839, 0.666, 0.669, 0.583, 0.168]])

        calc_result = find_consensus_motif(data, 5)
        bf_result = find_consensus_motif_bruteforce(data, 5)

        npt.assert_almost_equal(calc_result.radius, bf_result.radius)
        npt.assert_equal(calc_result.series_index, bf_result.series_index)
        npt.assert_equal(calc_result.subseq_index, bf_result.subseq_index)


def find_consensus_motif_bruteforce(series_list, m) -> CMResult:
    result = CMResult(np.inf, -1, -1)

    for series_idx, series in enumerate(series_list):
        radii = np.zeros(len(series) - m + 1)
        for series2_idx, series2 in enumerate(series_list):
            if series_idx == series2_idx:
                continue

            calc = AnytimeCalculator(m, series, series2)
            calc.add_generator(0, ZNormEuclidean())
            mp_cons = calc.add_consumer([0], MatrixProfileLR())
            calc.calculate_columns()
            mp = mp_cons.matrix_profile()

            radii = np.maximum(radii, mp)

        subseq_idx = np.argmin(radii)
        subseq_radius = radii[subseq_idx]
        if subseq_radius < result.radius:
            result = CMResult(subseq_radius, series_idx, subseq_idx)

    return result




