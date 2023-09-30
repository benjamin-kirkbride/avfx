from typing import List, Sequence, Tuple

import numpy as np


def erb_from_freq(freq: int) -> float:
    """Get equivalent rectangular bandwidth from the given frequency.

    See: https://en.wikipedia.org/wiki/Equivalent_rectangular_bandwidth

    Args:
        freq: input frequency

    Returns:
        int: cam
    """

    return float(9.265 * np.log(1 + np.divide(freq, 24.7 * 9.16)))


def generate_bin_edges(low_freq: int, high_freq: int, bin_count: int) -> List[int]:
    """Bin sizes designed around Equivalent Rectangular Bandwidth

    NOTE: this becomes less accurate as the bin values increase,
        but should be good enough (TM)

    Args:
        low_freq: where to start the bins (> 0)
        high_freq: where to end the bins (<= 20,000)
        count: number of bin edges to generate

    Returns:
        List[int]: bin separators in Hz
    """

    bin_edges = []
    cams = np.linspace(erb_from_freq(low_freq), erb_from_freq(high_freq), bin_count)
    for i, cam in enumerate(cams):
        if not i:
            # this is probably not nessesary, but better safe than sorry?
            bin_edges.append(low_freq)
        elif i == len(cams) - 1:
            bin_edges.append(high_freq)
        else:
            bin_edges.append(round(10 ** (cam / 21.4) / 0.00437 - 1 / 0.00437))

    return bin_edges


def generate_bins_from_edges(edges: List[int]) -> List[Tuple[int, int]]:
    bins = []
    previous_edge = edges[0]
    for edge in edges[1:]:
        bins.append((previous_edge, edge))
        previous_edge = edge
    return bins


def generate_edges_from_bin_centers(bin_centers: Sequence[int]) -> List[int]:
    # we assume for the first and last bin is symmetrical
    # (we actually assume they all are)
    # obviously no edge can be negative
    edges: List[int] = []

    previous_center = None
    for center in bin_centers:
        if previous_center is None:
            previous_center = center
            continue

        edge = previous_center + (center - previous_center) / 2
        # print(f"edge between {previous_center} and {center}: {edge}")
        if not edges:
            first_edge = previous_center - edge
            if first_edge < 0:
                first_edge = 0
            edges.append(first_edge)

        edges.append(edge)
        previous_center = center
    else:
        assert previous_center is not None
        # assume previous center truly is center
        final_edge = (previous_center - edges[-1]) + previous_center
        edges.append(final_edge)

    return edges


def map_bins_to_fft_indexes(
    actual_bins: list[tuple[int, int]], requested_bins: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    actual_bin_freq = [bin[1] for bin in actual_bins]  # Extract upper edge frequencies

    bin_indices = []
    for requested_bin in requested_bins:
        # Find the indexes where the upper edge frequency of actual_bin lies within the requested bin
        indices = [
            i
            for i, freq in enumerate(actual_bin_freq)
            if requested_bin[0] <= freq <= requested_bin[1]
        ]
        bin_indices.append(indices)

    bin_index_map = []
    for i in range(len(bin_indices)):
        if bin_indices[i]:  # non-empty current bin
            # When not at the last bin and the next bin is not empty
            if i + 1 < len(bin_indices) and bin_indices[i + 1]:
                bin_index_map.append((bin_indices[i][0], bin_indices[i + 1][0]))
            else:  # Last bin or next bin is empty
                bin_index_map.append((bin_indices[i][0], bin_indices[i][-1]))

    return bin_index_map


def get_bin_indexes(
    bin_edges: list[int],
    low_freq: int,
    high_freq: int,
    bin_count: int,
) -> List[Tuple[int, int]]:
    """Get the indexes of the bins that we want to use from the FFT result."""
    actual_bins = generate_bins_from_edges(bin_edges)

    requested_bin_edges = generate_bin_edges(
        low_freq=low_freq, high_freq=high_freq, bin_count=bin_count
    )
    requested_bins = generate_bins_from_edges(requested_bin_edges)

    bin_index_map = map_bins_to_fft_indexes(actual_bins, requested_bins)
    return bin_index_map
