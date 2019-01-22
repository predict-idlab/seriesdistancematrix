import numpy as np
from distancematrix.generator.znorm_euclidean import ZNormEuclidean
import time


def find_variable_length_motifs(series, min_motif_length, max_motif_length, cache_size=3, noise_std=0.):
    """
    Finds the top motif for each subsequence length in the given range. The top motif is defined as the
    subsequence (for a given length) for which the z-normalized euclidean distance is minimal, excluding any
    trivial matches.

    This method implements the VALMOD algorithm described in "Matrix Profile X: VALMOD - Scalable Discovery of
    Variable-Length Motifs in Data Series" by M. Linardi et al.

    :param series: one dimensional time series
    :param min_motif_length: minimum motif length
    :param max_motif_length: maximum motif length (inclusive)
    :param cache_size: number of entries kept in memory per subsequence (can only affect performance, default should
      be okay)
    :param noise_std: standard deviation of noise on the signal, used for correcting the z-normalized euclidean distance
    :return: a list of tuples of length (max_motif_length - min_motif_length + 1), containing the indices of the
      motif and its match
    """

    if series.ndim != 1:
        raise RuntimeError("Series should be 1D")
    if min_motif_length < 2 or not np.isfinite(min_motif_length):
        raise RuntimeError("Invalid min_motif_length: " + str(min_motif_length))
    if max_motif_length < min_motif_length or not np.isfinite(max_motif_length):
        raise RuntimeError("Invalid max_motif_length: " + str(min_motif_length))
    if cache_size < 0:
        raise RuntimeError("Invalid p: " + str(min_motif_length))

    # Stores for each motif length a tuple of the indices of the motif
    motifs_found = []

    dist_generator = ZNormEuclidean(noise_std=noise_std).prepare(min_motif_length, series)

    # Full distance matrix calculation for first motif length
    lb_lists, best_motif_idxs = _find_all_motifs_full_matrix_iteration(dist_generator, cache_size,
                                                                       int(np.ceil(min_motif_length / 2)))
    motifs_found.append(best_motif_idxs)

    # For all following motif lengths: try exploiting the lower bound to avoid calculations
    for m in range(min_motif_length + 1, max_motif_length + 1):
        # Note: might be possible to simply update the existing generator?
        dist_generator = ZNormEuclidean(noise_std=noise_std).prepare(m, series)

        num_subseq = len(series) - m + 1
        trivial_match_buffer = int(np.ceil(min_motif_length / 2))

        best_candidate_motif_distance = np.inf
        best_candidate_motif_idxs = None
        invalid_subseq_idxs = []  # Indices of subsequences for which lower bound pruning did not work
        invalid_subseq_lbs = []  # Lower bound for the match on subsequences where pruning did not work

        for i in range(num_subseq):
            subseq_lb_list = lb_lists[i]

            best_match_entry = None
            best_match_distance = np.inf
            subseq_lower_bound = -1

            for entry in subseq_lb_list:
                # As motif length grows, some lowerbound entries may have become trivial matches
                if abs(entry.q_index - i) <= trivial_match_buffer:
                    continue

                # Or they may no longer contain valid indices
                if entry.q_index >= num_subseq or entry.s_index >= num_subseq:
                    continue

                entry.dot_prod += series[entry.q_index + m - 1] * series[entry.s_index + m - 1]

                # Calculate actual distance for these indices
                dist = dist_generator.calc_single(entry.q_index, entry.s_index, dot_prod=entry.dot_prod)
                if dist < best_match_distance:
                    best_match_distance = dist
                    best_match_entry = entry

                # Calculate lower bound using last (highest) non-trivial entry
                # (all previous entries should have lower bound)
                subseq_lower_bound = max(subseq_lower_bound, entry.lower_bound_base / dist_generator.std_s[i])

            # if minimum of actual distances < largest lower bound
            if best_match_distance < subseq_lower_bound:
                # best match for this subseq found
                if best_match_distance < best_candidate_motif_distance:
                    best_candidate_motif_distance = best_match_distance
                    best_candidate_motif_idxs = (best_match_entry.q_index, best_match_entry.s_index)
            else:
                # best match may be outside the lowerbound entries, but we have a lower bound for its distance
                invalid_subseq_idxs.append(i)
                invalid_subseq_lbs.append(subseq_lower_bound)

        # If the best candidate motif has a lower distance than all lower bounds, we have the motif
        if best_candidate_motif_idxs and best_candidate_motif_distance <= np.min(invalid_subseq_lbs):
            motifs_found.append(best_candidate_motif_idxs)
            continue

        # if not, we need to calculate all those whose lower bound was lower than the candidate motif to be sure
        if len(invalid_subseq_idxs) > num_subseq * np.log(num_subseq):
            # If too many columns have to be recalculated, recalculate the entire matrix and update the lb_lists.
            # A clear boundary for when this should happen isn't available,
            # different strategies might affect performance (but not correctness)
            lb_lists, best_candidate_motif_idxs = _find_all_motifs_full_matrix_iteration(
                dist_generator, cache_size, trivial_match_buffer)
        else:
            # Recalculate all columns that might have a better match
            for invalid_idx, lower_bound in zip(invalid_subseq_idxs, invalid_subseq_lbs):
                if lower_bound < best_candidate_motif_distance:
                    distances = dist_generator.calc_column(invalid_idx)
                    trivial_match_start = max(0, invalid_idx - trivial_match_buffer)
                    trivial_match_end = invalid_idx + trivial_match_buffer + 1
                    distances[trivial_match_start: trivial_match_end] = np.inf
                    best_match_distance = np.min(distances)

                    if best_match_distance < best_candidate_motif_distance:
                        best_candidate_motif_distance = best_match_distance
                        best_candidate_motif_idxs = (np.argmin(distances), invalid_idx)

        # We now have the best motif for sure
        motifs_found.append(best_candidate_motif_idxs)

    return motifs_found


def _find_all_motifs_full_matrix_iteration(dist_generator, lb_list_size, trivial_match_buffer):
    """
    Calculates the entire distance matrix using the provided distance generator.
    For each column, lower bounds are calculated (as described in the VALMOD paper) and the lb_list_size best entries
    are stored (ordered by ascending distance).

    :param dist_generator: z-normalized distance generator
    :param lb_list_size: max number of lower bound entries to store
    :param trivial_match_buffer: trivial match buffer, the lb_list will not contain any entries that fall inside
      this buffer
    :return: tuple of: list of all lb_lists per column, indices of the best motif for the entire distance matrix
    """
    num_subseq = dist_generator.mu_s.view.shape[0]
    subseq_length = dist_generator.m

    lb_lists = []
    best_motif_dist = np.Inf
    best_motif_idxs = None

    for column_idx in range(num_subseq):
        distances = dist_generator.calc_column(column_idx)

        # Find best match, while avoiding trivial matches
        trivial_match_start = max(0, column_idx - trivial_match_buffer)
        trivial_match_end = column_idx + trivial_match_buffer + 1
        distances[trivial_match_start: trivial_match_end] = np.inf

        best_dist = np.min(distances)
        if best_dist < best_motif_dist:
            best_motif_dist = best_dist
            best_motif_idxs = (np.argmin(distances), column_idx)

        # Determine lower boundaries
        dotprod = dist_generator.prev_calc_column_dot_prod
        mu = dist_generator.mu_s.view
        std = dist_generator.std_s.view

        if std[column_idx] == 0:
            # In case one of the stds is zero, there is no defined formula for a lower bound (not found yet at least).
            # So we simply return no empty bounds, so this column will always be calculated.
            lb_list = []
            # We can get away with only checking std[column_idx] and not every entry of std (in the else clause):
            # if a lower bound is underestimated, it can only result in unneeded calculation, which is ok
            # if a lower bound is overestimated, a motif for a stable signal may go undetected, but since the entire
            # column will be calculated, it will be found this way.
        else:
            lower_bound_q = np.clip((dotprod / subseq_length - mu * mu[column_idx]) / (std * std[column_idx]), 0, 1)
            lower_bound_base = np.sqrt(subseq_length * (1 - np.square(lower_bound_q))) * std[column_idx]
            lower_bound_base[trivial_match_start: trivial_match_end] = np.inf

            closest_indices = np.argsort(lower_bound_base)[:lb_list_size]

            # Cover corner case where there may not be enough non-trivial matches to fill the lb_list
            if lower_bound_base[closest_indices[-1]] == np.inf:
                first_inf_idx = np.searchsorted(lower_bound_base[closest_indices], np.inf)
                closest_indices = closest_indices[:first_inf_idx]

            lb_list = []
            for i in range(len(closest_indices)):
                lb_list.append(LowerBoundEntry(closest_indices[i], column_idx, lower_bound_base[closest_indices[i]],
                                               dotprod[closest_indices[i]]))
        lb_lists.append(lb_list)

    return lb_lists, best_motif_idxs


class LowerBoundEntry:
    def __init__(self, q_index, s_index, lower_bound_base, dot_prod):
        self.q_index = q_index
        self.s_index = s_index
        self.lower_bound_base = lower_bound_base
        self.dot_prod = dot_prod
