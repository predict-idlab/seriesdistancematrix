from collections import namedtuple
import numpy as np

from distancematrix import AnytimeCalculator
from distancematrix.generator import ZNormEuclidean
from distancematrix.consumer import MatrixProfileLR

CMResult = namedtuple('CMResult', ['radius', 'series_index', 'subseq_index'])


def find_consensus_motif(series_list, m: int) -> CMResult:
    """
    Finds the consensus motif and corresponding distance for the given collection of series.
    The consensus motif is the subsequence (extracted from one of the series),
    that has a match to a subsequence from each other series within a certain distance,
    where that distance is minimal.

    This method implements the Ostinato algorithm, described in
    "Matrix Profile XV: Exploiting Time Series Consensus  Motifs to Find Structure in Time Series Sets"
    by K. Kamgar, S. Gharghabi and E. Keogh.

    :param series_list: list of 1-dimensional arrays
    :param m: length of the consensus motif
    :return: tuple containing radius, series index and subsequence index of the consensus motif
    """
    if len(series_list) < 2:
        raise RuntimeError("At least 2 series are required.")
    if m < 3:
        raise RuntimeError("Motif length should be >= 3.")
    for series in series_list:
        series = np.array(series)
        if len(series) < m:
            raise RuntimeError("One or more series are shorter than the desired motif length.")
        if series.ndim != 1:
            raise RuntimeError("One or more series are not one dimensional.")

    best_result = CMResult(np.inf, -1, -1)
    num_series = len(series_list)

    # Using streaming generators avoids having to recalculate the means/stds for calculating
    # distance between the series and a single subsequence
    cached_generators = []
    for series in series_list:
        gen = ZNormEuclidean().prepare_streaming(m, m, len(series))
        gen.append_query(series)
        cached_generators.append(gen)

    for series_idx in range(num_series):
        next_series_idx = (series_idx + 1) % num_series
        active_series = series_list[series_idx]

        mp = _calculate_mp(m, series_list[series_idx], series_list[next_series_idx])
        candidates = np.argsort(mp)

        # Iterate over all candidate subsequences, starting from those that had the best match to next_series.
        for subseq_idx in candidates:
            candidate_radius = mp[subseq_idx]
            aborted = False

            if candidate_radius >= best_result.radius:
                break

            for other_series_idx in range(num_series):
                if other_series_idx in [series_idx, next_series_idx]:
                    continue

                # Calculates the distance from the candidate subsequence to all subsequences in other_series.
                other_gen = cached_generators[other_series_idx]
                other_gen.append_series(active_series[subseq_idx: subseq_idx+m])
                distances = other_gen.calc_column(0)
                min_distance = np.min(distances)
                candidate_radius = max(candidate_radius, min_distance)

                if candidate_radius >= best_result.radius:
                    aborted = True
                    break

            if not aborted and candidate_radius < best_result.radius:
                best_result = CMResult(candidate_radius, series_idx, subseq_idx)

    return best_result


def _calculate_mp(m, series, query) -> np.array:
    """Calculates the z-norm-based Matrix Profile."""

    # Todo: MP_LR will have unneeded overhead, change to lightweight MP (only MP, no idx, no left/right)
    calc = AnytimeCalculator(m, series, query)
    calc.add_generator(0, ZNormEuclidean())
    cons = calc.add_consumer([0], MatrixProfileLR())
    calc.calculate_columns()
    return cons.matrix_profile()
