"""
Standalone fiber-seq utility functions. All functions return numpy arrays.
No torch dependency — safe to import anywhere, including outside training loops.
"""

import os
import sys
import contextlib
import numpy as np


#--------------------------------------------------------------------------------------------------
# I/O Suppression

@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr at the OS level."""
    devnull = os.open(os.devnull, os.O_RDWR)
    save_stdout = os.dup(1)
    save_stderr = os.dup(2)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)
        os.close(save_stdout)
        os.close(save_stderr)
        os.close(devnull)


#--------------------------------------------------------------------------------------------------
# DNA Utilities

def _dna_to_onehot(sequence):
    """Convert a nucleotide string to (L, 4) float32 array. N/unknown → all-zeros row."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    out = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, nuc in enumerate(sequence.upper()):
        j = mapping.get(nuc, -1)
        if j >= 0:
            out[i, j] = 1.0
    return out


def get_locus_onehot(fasta, chrom, start, end):
    """Fetch genomic sequence and return (context_length, 4) float32 numpy array."""
    seq = fasta.fetch(chrom, start, end)
    return _dna_to_onehot(seq)


#--------------------------------------------------------------------------------------------------
# Fiber Signal Extraction
# All functions return np.ndarray of shape (context_length,) where context_length = end - start.
# Overlap-aware: the -1 background mask is set only within the fiber's actual alignment coverage.

def get_m6a(fiber, start, end, ref_dna_seq, Q_THRESHOLD=200):
    context_length = end - start
    m6a_data = np.zeros(context_length, dtype=np.float32)

    overlap_start = max(start, fiber.start)
    overlap_end   = min(end,   fiber.end)
    if overlap_start >= overlap_end:
        return m6a_data

    f_start_idx = overlap_start - start
    f_end_idx   = overlap_end   - start

    ref_seq_arr = np.array(list(ref_dna_seq.upper()))
    at_mask = np.isin(ref_seq_arr, ['A', 'T'])

    fiber_at_mask = np.zeros(context_length, dtype=bool)
    fiber_at_mask[f_start_idx:f_end_idx] = at_mask[f_start_idx:f_end_idx]
    m6a_data[fiber_at_mask] = -1.0

    ref_starts = np.array(fiber.m6a.reference_starts, dtype=np.float32)
    qualities  = np.array(fiber.m6a.ml, dtype=np.float32)

    mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)
    valid_positions = (ref_starts[mask] - start).astype(np.int32)
    m6a_data[valid_positions] = 1.0
    return m6a_data


def get_cpg(fiber, start, end, ref_dna_seq, Q_THRESHOLD=200):
    context_length = end - start
    cpg_data = np.zeros(context_length, dtype=np.float32)

    overlap_start = max(start, fiber.start)
    overlap_end   = min(end,   fiber.end)

    if overlap_start < overlap_end:
        local_start = int(overlap_start - start)
        local_end   = int(overlap_end   - start)

        ref_seq_arr = np.array(list(ref_dna_seq.upper()))

        c_positions = np.where(ref_seq_arr[:-1] == 'C')[0]
        valid_cg    = c_positions[ref_seq_arr[c_positions + 1] == 'G']

        g_positions = np.where(ref_seq_arr[1:] == 'G')[0] + 1
        valid_gc    = g_positions[ref_seq_arr[g_positions - 1] == 'C']

        all_cpg_indices  = np.unique(np.concatenate([valid_cg, valid_gc]))
        in_fiber_mask    = (all_cpg_indices >= local_start) & (all_cpg_indices < local_end)
        cpg_data[all_cpg_indices[in_fiber_mask]] = -1.0

    ref_starts = np.array(fiber.cpg.reference_starts, dtype=np.float32)
    qualities  = np.array(fiber.cpg.ml, dtype=np.float32)

    mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)
    valid_positions = (ref_starts[mask] - start).astype(np.int32)
    cpg_data[valid_positions] = 1.0
    return cpg_data


def get_msp(fiber, start, end, ref_dna_seq, Q_THRESHOLD=0):
    context_length = end - start
    msp_data = np.zeros(context_length, dtype=np.float32)

    for ref_pos, length, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
        if ref_pos is None or length is None:
            continue
        ref_end = ref_pos + length
        if ref_pos < end and ref_end > start and aq >= Q_THRESHOLD:
            win_start = max(0, ref_pos - start)
            win_end   = min(context_length, ref_end - start)
            msp_data[win_start:win_end] = 1

    return msp_data


def get_nuc(fiber, start, end, ref_dna_seq, Q_THRESHOLD=0):
    context_length = end - start
    nuc_data = np.zeros(context_length, dtype=np.float32)

    for ref_pos, length, aq in zip(fiber.nuc.reference_starts, fiber.nuc.reference_lengths, fiber.nuc.qual):
        if ref_pos is None or length is None:
            continue
        ref_end = ref_pos + length
        if ref_pos < end and ref_end > start and aq >= Q_THRESHOLD:
            win_start = max(0, ref_pos - start)
            win_end   = min(context_length, ref_end - start)
            nuc_data[win_start:win_end] = 1

    return nuc_data


def get_fire_msp(fiber, start, end, ref_dna_seq, Q_THRESHOLD=200):
    context_length = end - start
    fire_msp_data = np.zeros(context_length, dtype=np.float32)

    fire_source = getattr(fiber, 'fire_msp', fiber.msp)

    for ref_pos, length, aq in zip(fire_source.reference_starts, fire_source.reference_lengths, fire_source.qual):
        if ref_pos is None or length is None:
            continue
        ref_end = ref_pos + length
        if ref_pos < end and ref_end > start and aq >= Q_THRESHOLD:
            win_start = max(0, ref_pos - start)
            win_end   = min(context_length, ref_end - start)
            fire_msp_data[win_start:win_end] = 1

    return fire_msp_data


#--------------------------------------------------------------------------------------------------
# Fiber Batch Collection

def get_fiber_data(
    fiber_bam, chrom, start, end, fasta,
    fibers_per_entry, context_length, input_features,
    return_fiber_dna=False, min_overlap=50,
):
    """
    Collect fiber reads over [chrom:start-end] and extract per-fiber feature arrays.

    Parameters
    ----------
    fiber_bam       : open pyft.Fiberbam handle
    chrom, start, end : genomic locus
    fasta           : open pysam.FastaFile handle
    fibers_per_entry : max fibers to collect (N dimension)
    context_length  : window size in bp (= end - start)
    input_features  : list of signal functions [fn(fiber, start, end, ref_dna_seq) -> (L,)]
    return_fiber_dna : whether to build per-fiber DNA one-hot array
    min_overlap     : minimum bp overlap between fiber alignment and window

    Returns
    -------
    fibers_array   : np.ndarray  (fibers_per_entry, n_features, context_length)
    dna_array      : np.ndarray  (fibers_per_entry, context_length, 4)  or  None
    n_fibers       : int  — actual fibers collected (≤ fibers_per_entry)
    fiber_coverage : np.ndarray  (context_length,)  — per-position fiber count
    """
    ref_dna_seq    = fasta.fetch(chrom, start, end)
    fibers_array   = np.zeros((fibers_per_entry, len(input_features), context_length), dtype=np.float32)
    dna_array      = np.zeros((fibers_per_entry, context_length, 4), dtype=np.float32) if return_fiber_dna else None
    fiber_coverage = np.zeros(context_length, dtype=np.float32)

    with suppress_stdout_stderr():
        possible_fibers = fiber_bam.fetch(chrom, start, end)

    n_fibers = 0
    for fiber in possible_fibers:
        if n_fibers == fibers_per_entry:
            break

        overlap_start = max(start, fiber.start)
        overlap_end   = min(end,   fiber.end)
        if overlap_end - overlap_start < min_overlap:
            continue

        win_start = overlap_start - start
        win_end   = overlap_end   - start
        fiber_coverage[win_start:win_end] += 1

        if return_fiber_dna:
            dna_buffer = list("N" * context_length)
            read_off   = overlap_start - fiber.start
            read_end   = overlap_end   - fiber.start
            if fiber.seq is not None:
                slc     = fiber.seq[read_off:read_end]
                slc_len = min(len(slc), context_length - win_start)
                if slc_len > 0:
                    dna_buffer[win_start:win_start + slc_len] = list(slc[:slc_len])
            dna_array[n_fibers] = _dna_to_onehot("".join(dna_buffer))

        fibers_array[n_fibers] = np.array([fn(fiber, start, end, ref_dna_seq) for fn in input_features])
        n_fibers += 1

    return fibers_array, dna_array, n_fibers, fiber_coverage
