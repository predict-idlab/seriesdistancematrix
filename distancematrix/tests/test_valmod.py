import unittest

import numpy as np
from unittest import TestCase
import numpy.testing as npt

from distancematrix.valmod import _find_all_motifs_full_matrix_iteration
from distancematrix.valmod import LowerBoundEntry
from distancematrix.generator.znorm_euclidean import ZNormEuclidean


class TestValmod(TestCase):
    def _test_find_all_motifs_full_matrix_iteration(self, data, m, lb_list_size):
        dist_gen = ZNormEuclidean(0.).prepare(m, data)

        calc_lb_lists, calc_motif_idxs = _find_all_motifs_full_matrix_iteration(dist_gen, lb_list_size, int(np.ceil(m / 2)))
        bf_lb_lists, bf_motif_idxs = bruteforce_full_matrix_iteration(data, m, lb_list_size)

        npt.assert_equal(set(bf_motif_idxs), set(calc_motif_idxs))

        npt.assert_equal(len(bf_lb_lists), len(calc_lb_lists))
        for iteration, (bf_lb_list, calc_lb_list) in enumerate(zip(bf_lb_lists, calc_lb_lists)):
            # Ensure lower bounds match
            bf_lower_bounds = [e.lower_bound_base for e in bf_lb_list]
            calc_lower_bounds = [e.lower_bound_base for e in calc_lb_list]

            npt.assert_allclose(bf_lower_bounds, calc_lower_bounds, err_msg="Mismatch for iteration " + str(iteration))

            if len(bf_lower_bounds) == 0:
                continue

            # Since multiple entries may have the same lower bound for different dot products: sort again
            bf_lb_list.sort(key=lambda e: (e.lower_bound_base, e.dot_prod, e.q_index))
            calc_lb_list.sort(key=lambda e: (e.lower_bound_base, e.dot_prod, e.q_index))
            lists_match_upto = bf_lower_bounds.index(bf_lower_bounds[-1])

            npt.assert_allclose(
                [e.dot_prod for e in bf_lb_list[:lists_match_upto]],
                [e.dot_prod for e in calc_lb_list[:lists_match_upto]])

            npt.assert_equal(
                [(e.q_index, e.s_index) for e in bf_lb_list[:lists_match_upto]],
                [(e.q_index, e.s_index) for e in calc_lb_list[:lists_match_upto]])

            for entry in calc_lb_list[lists_match_upto:]:
                subseq_1 = data[entry.q_index: entry.q_index + m ]
                subseq_2 = data[entry.s_index: entry.s_index + m]
                npt.assert_almost_equal(np.sum(subseq_1 * subseq_2), entry.dot_prod)

    def test_find_all_motifs_full_matrix_iteration_normal_data(self):
        # Random data, 20 points
        data = np.array(
            [-1.61, -0.43, -0.43, 0.82, 0.42, 1.58, -0.46, 1.41, 1.31,
             -0.13, -0.05, 0.59, 1.76, -0.43, -0.14, -0.14, 1.07, 1.1, 0.84, -1.49])

        self._test_find_all_motifs_full_matrix_iteration(data, 4, 1)
        self._test_find_all_motifs_full_matrix_iteration(data, 4, 5)

        # Due to the large subseq length, some lower bound arrays will be empty
        self._test_find_all_motifs_full_matrix_iteration(data, 10, 5)

    # Because the division by zero can result in inf or -inf, results for the lower bound are not deterministic,
    # which is a pain to test. Behavior should be correct though.
    @unittest.skip("VALMOD: Flat signals have undefined lower bounds.")
    def test_find_all_motifs_full_matrix_iteration_data_with_flats(self):
        # Random data, 20 points, with flat signals
        data = np.array(
            [-1.61, -0.43, -0.43, -0.43, -0.43, 1.58, -0.46, 1.41, 1.31,
             -0.13, -0.05, 0.59, 1.76, -0.43, 0.84, 0.84, 0.84, 0.84, 0.84, -1.49])

        self._test_find_all_motifs_full_matrix_iteration(data, 4, 1)
        self._test_find_all_motifs_full_matrix_iteration(data, 4, 5)

        # Due to the large subseq length, some lower bound arrays will be empty
        self._test_find_all_motifs_full_matrix_iteration(data, 10, 5)


def bruteforce_full_matrix_iteration(series, subseq_length, lb_list_size):
    """
    Brute force implementation of _find_all_motifs_full_matrix_iteration

    :param series: 1D series
    :param subseq_length: subsequence length to use
    :param lb_list_size: max size of lower bound lists
    :return: tuple of: list of all lb_lists per column, indices of the best motif for the entire distance matrix
    """
    num_subseq = series.shape[0] - subseq_length + 1
    triv_match_buffer = int(np.ceil(subseq_length / 2))

    means = np.array([np.mean(series[i: i + subseq_length]) for i in range(num_subseq)])
    stds = np.array([np.std(series[i: i + subseq_length]) for i in range(num_subseq)])

    # Finding the best motif
    motif_dist2 = np.inf
    motif_idxs = None

    # Lower bounds
    lb_lists = []

    for s_i in range(num_subseq):
        subseq_1 = series[s_i: s_i + subseq_length]

        lb_list = []

        for q_i in range(num_subseq):
            # Avoid trivial match
            if abs(s_i - q_i) <= triv_match_buffer:
                continue

            subseq_2 = series[q_i: q_i + subseq_length]
            dot_prod = np.sum(subseq_1 * subseq_2)

            # Calculate z-normalised distance (squared)
            if stds[s_i] == 0 and stds[q_i] == 0:
                z_dist2 = 0
            elif stds[s_i] == 0 or stds[q_i] == 0:
                z_dist2 = np.square(subseq_length)
            else:
                z_dist2 = 2 * (subseq_length - (dot_prod - subseq_length * means[s_i] * means[q_i]) /
                               (stds[s_i] * stds[q_i]))

            if z_dist2 < motif_dist2:
                motif_dist2 = z_dist2
                motif_idxs = (s_i, q_i)

            # Calculate lower bound
            if stds[s_i] != 0:
                std_q = stds[q_i]
                lower_bound_q = np.clip(
                    (dot_prod / subseq_length - means[s_i] * means[q_i]) / (stds[s_i] * std_q), 0, 1)
                lower_bound = np.sqrt(subseq_length * (1 - np.square(lower_bound_q))) * stds[s_i]

                lb_list.append(LowerBoundEntry(q_i, s_i, lower_bound, dot_prod))

        # Trim lower bound lists
        lb_lists.append(sorted(lb_list, key=lambda e: e.lower_bound_base)[:lb_list_size])

    return lb_lists, motif_idxs