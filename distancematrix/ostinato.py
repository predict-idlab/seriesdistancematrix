from collections import namedtuple
import numpy as np

from distancematrix import AnytimeCalculator
from distancematrix.generator import ZNormEuclidean
from distancematrix.generator.znorm_euclidean import BoundZNormEuclidean, _CONSTANT_SUBSEQ_THRESHOLD
from distancematrix.consumer import MatrixProfileLR
from distancematrix.math_tricks import sliding_mean_std
from distancematrix.ringbuffer import RingBuffer

CMResult = namedtuple('CMResult', ['radius', 'series_index', 'subseq_index'])


def find_consensus_motif(series_list, m: int) -> CMResult:
    """
    Finds the top-1 consensus motif and corresponding distance for the given collection of series.
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

    # Create a distance calculator for each series pair, but reuse mu/std calculations per series.
    # Step 1: mu/std calculation
    cached_generators = {}
    mus = []
    stds = []
    stdsz = []
    for series in series_list:
        mu, std = sliding_mean_std(series, m)
        mus.append(RingBuffer(mu, scaling_factor=1.))
        stds.append(RingBuffer(std, scaling_factor=1.))
        stdsz.append(RingBuffer(std > _CONSTANT_SUBSEQ_THRESHOLD, scaling_factor=1.))

    # Step 2: create the distance calculator
    for i, series1 in enumerate(series_list):
        for j, series2 in enumerate(series_list):
            if i == j:
                continue
            gen = BoundZNormEuclidean(m, RingBuffer(series1, scaling_factor=1.), RingBuffer(series2, scaling_factor=1.),
                                      False, 0., mus[i], stds[i], stdsz[i], mus[j], stds[j], stdsz[j])
            cached_generators[i, j] = gen

    # Look for the consensus motif: iterator over all series
    for series_idx in range(num_series):
        next_series_idx = (series_idx + 1) % num_series
        active_series = series_list[series_idx]

        # Calculate a full matrix profile between the series and the next series
        dist_calc = cached_generators[(series_idx, next_series_idx)]
        num_subseq = len(active_series) - m + 1
        mp = np.empty(num_subseq, dtype=float)
        for col in range(num_subseq):
            mp[col] = np.min(dist_calc.calc_column(col))

        # Order the subsequences of the series from lowest to highest distances (as given by the Matrix Profile)
        candidates = np.argsort(mp)

        # Iterate over all candidate subsequences, starting from those that had the best match to next_series.
        for subseq_idx in candidates:
            candidate_radius = mp[subseq_idx]
            aborted = False

            # Abort if the distance (to next_series) is worse than best result so far
            if candidate_radius >= best_result.radius:
                break

            # Check distance of the candidate subsequence to all other series.
            for other_series_idx in range(num_series):
                # Skip the current and next_series, as we already considered those.
                if other_series_idx in [series_idx, next_series_idx]:
                    continue

                # Calculates the distance from the candidate subsequence to all subsequences in other_series.
                other_gen = cached_generators[(series_idx, other_series_idx)]
                distances = other_gen.calc_column(subseq_idx)
                min_distance = np.min(distances)
                candidate_radius = max(candidate_radius, min_distance)

                # Abort search if distance is greater than best so far.
                if candidate_radius >= best_result.radius:
                    aborted = True
                    break

            # Store the current candidate as best result so far.
            if not aborted and candidate_radius < best_result.radius:
                best_result = CMResult(candidate_radius, series_idx, subseq_idx)

    return best_result


def find_consensus_motif_subset(series_list, m: int, k: int) -> CMResult:
    """
    Finds the top-1 k of n consensus motif and corresponding distance for the given collection of series.
    The consensus motif is the subsequence (extracted from one of the series),
    that has a match to a subsequence from k other series within a certain distance,
    where that distance is minimal.

    This method implements the k of n Ostinato algorithm, described in
    "Matrix Profile XV: Exploiting Time Series Consensus  Motifs to Find Structure in Time Series Sets"
    by K. Kamgar, S. Gharghabi and E. Keogh.

    Note: this algorithm has not yet been optimized for speed.
    (Instead, consider using the Anytime Ostinato algorithm.)

    :param series_list: list of 1-dimensional arrays
    :param m: length of the consensus motif
    :return: tuple containing radius, series index and subsequence index of the consensus motif
    """
    if len(series_list) < 2:
        raise RuntimeError("At least 2 series are required.")
    if m < 3:
        raise RuntimeError("Motif length should be >= 3.")
    if k < 2 or k > len(series_list):
        raise RuntimeError("Number of considered series should be >= 2 and <= len(series).")
    for series in series_list:
        series = np.array(series)
        if len(series) < m:
            raise RuntimeError("One or more series are shorter than the desired motif length.")
        if series.ndim != 1:
            raise RuntimeError("One or more series are not one dimensional.")

    best_result = CMResult(np.inf, -1, -1)
    num_series = len(series_list)
    num_ignored_series = num_series - k

    # Using streaming generators avoids having to recalculate the means/stds for calculating
    # distance between the series and a single subsequence
    cached_generators = []
    for series in series_list:
        gen = ZNormEuclidean().prepare_streaming(m, m, len(series))
        gen.append_query(series)
        cached_generators.append(gen)

    for series_idx in range(num_series):
        active_series = series_list[series_idx]
        num_subseqs = len(active_series) - m + 1

        # Calculate for each subsequence in active_series the best match to the next
        # (num_ignored_series + 1) series
        next_mps = np.empty((num_ignored_series + 1, num_subseqs))
        for i in range(num_ignored_series + 1):
            next_series_idx = (series_idx + 1 + i) % num_series
            next_series = series_list[next_series_idx]
            next_mps[i, :] = _calculate_mp(m, active_series, next_series)

        candidates = np.argsort(np.min(next_mps, axis=0))

        # Iterate over all candidate subsequences, starting from those that had the best match to any next_series.
        for subseq_idx in candidates:
            aborted = False

            # We track the (num_ignored_series + 1) biggest radii found,
            # where only the smallest value determines the actual radius for the subsequence
            # (since we can ignore the other values).
            candidate_radii: np.ndarray = next_mps[:, subseq_idx].copy()

            # Iterate over all other, not yet calculated, series
            for j in range(num_series - num_ignored_series - 2):
                other_series_idx = (series_idx + num_ignored_series + 2) % num_series

                candidate_radii.sort()
                if candidate_radii[0] >= best_result.radius:
                    aborted = True
                    break

                # Calculates the distance from the candidate subsequence to all subsequences in other_series.
                other_gen = cached_generators[other_series_idx]
                other_gen.append_series(active_series[subseq_idx: subseq_idx + m])
                min_distance = np.min(other_gen.calc_column(0))
                candidate_radii[0] = max(candidate_radii[0], min_distance)

            if not aborted:
                best_radius = np.min(candidate_radii)
                if best_radius < best_result.radius:
                    best_result = CMResult(best_radius, series_idx, subseq_idx)

    return best_result


def _calculate_mp(m, series, query) -> np.array:
    """Calculates the z-norm-based Matrix Profile."""

    # Todo: MP_LR will have unneeded overhead, change to lightweight MP (only MP, no idx, no left/right)
    calc = AnytimeCalculator(m, series, query)
    calc.add_generator(0, ZNormEuclidean())
    cons = calc.add_consumer([0], MatrixProfileLR())
    calc.calculate_columns()
    return cons.matrix_profile()


class _MPReverse(MatrixProfileLR):
    def __init__(self):
        super().__init__()

    def initialise(self, dims, query_subseq, series_subseq):
        super().initialise(dims, series_subseq, query_subseq)

    def process_diagonal(self, diag, values):
        super().process_diagonal(-diag, values)


class OstinatoAnytime(object):
    """
    Implementation of the Anytime Ostinato algorithm, which can be used to find the radius profile
    for a collection of series. Since it is an anytime algorithm, the user can choose between more accurate results
    or a shorter runtime.

    The radius profile contains for each subsequence the minimum distance needed to match a subsequence
    from all other series.
    Given the radius profile, the top-k minimal values correspond to the top-k consensus motifs.

    This algorithm is described in
    "Mining Recurring Patterns in Real-Valued Time Series using the Radius Profile"
    by D. De Paepe and S. Van Hoecke.
    """
    def __init__(self, series, m: int) -> None:
        """
        Creates a new instance that can be used to find the radius profile for the given series.

        :param series: the series for which to calculate the radius profile, a list of 1-D series
        :param m: subsequence length
        """
        num_series = len(series)

        self.calculators = []
        self.mps = [[] for i in range(num_series)]

        for i in range(num_series):
            for j in range(i + 1, num_series):
                calc = AnytimeCalculator(m, series[j], series[i])
                calc.add_generator(0, ZNormEuclidean())

                self.mps[j].append(calc.add_consumer([0], MatrixProfileLR()))
                self.mps[i].append(calc.add_consumer([0], _MPReverse()))
                self.calculators.append(calc)

    def calculate(self, fraction: float):
        """
        Calculates a given fraction of all distances.

        Experiments show that even for low fractions, the resulting radius profile will give representative
        approximate results. The runtime of this method scales linear with the fraction.

        :param fraction: fraction of values to calculate, value in [0 .. 1]
        """
        for calc in self.calculators:
            calc.calculate_diagonals(fraction)

    def get_radii(self, k_best: int = None):
        """
        Retrieves the radius profile for each series.
        If the calculation was not performed completely, the returned profiles will overestimate the real
        radius profile.

        :param k_best: If specified, calculates the radius using only the k_best best matching series
         (instead of all series)
        """
        radii = []

        for serie_consumers in self.mps:
            serie_mps = [cons.matrix_profile() for cons in serie_consumers]

            if k_best is None:
                radii.append(np.max(serie_mps, axis=0))
            else:
                radii.append(np.sort(serie_mps, axis=0)[k_best-1, :])

        return radii
