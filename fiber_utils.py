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

# Byte -> row of _ONEHOT_ROWS: A/C/G/T (either case) -> 0..3, anything else -> 4 (all-zeros row)
_ONEHOT_INDEX = np.full(256, 4, dtype=np.intp)
for _i, _base in enumerate(b"ACGT"):
    _ONEHOT_INDEX[_base] = _ONEHOT_INDEX[_base + 32] = _i
_ONEHOT_ROWS = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), dtype=np.float32)])


def _dna_to_onehot(sequence):
    """Convert a nucleotide string to (L, 4) float32 array. N/unknown → all-zeros row."""
    return _ONEHOT_ROWS[_ONEHOT_INDEX[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]]


def get_locus_onehot(fasta, chrom, start, end):
    """Fetch genomic sequence and return (context_length, 4) float32 numpy array."""
    seq = fasta.fetch(chrom, start, end)
    return _dna_to_onehot(seq)


#--------------------------------------------------------------------------------------------------
# Fiber Signal Extraction
# All functions take (fiber, start, end, ref) with ref a RefWindow for [start, end), and return
# np.ndarray of shape (context_length,) where context_length = end - start.
# Overlap-aware: the -1 background mask is set only within the fiber's actual alignment coverage.

class RefWindow:
    """Reference sequence of one window plus the masks every fiber in it shares, computed once per window
    (not once per fiber): at_mask marks A/T bases (m6a sites), cpg_mask marks both bases of each CpG."""
    def __init__(self, seq):
        self.seq = seq
        b = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
        self.at_mask = (b == ord("A")) | (b == ord("T"))
        cg = np.flatnonzero((b[:-1] == ord("C")) & (b[1:] == ord("G")))
        self.cpg_mask = np.zeros(len(b), dtype=bool)
        self.cpg_mask[cg] = True
        self.cpg_mask[cg + 1] = True


def _call_positions(mods, start, end, q_threshold):
    """Window-relative positions of base-modification calls in [start, end) with ML >= q_threshold.
    Positions stay int64: float32 cannot hold genomic coordinates above 2**24 (16.7 Mb) exactly, which used to
    shift calls by 1-2 bp. Calls without a reference position (None) are dropped."""
    starts = mods.reference_starts
    try:
        pos = np.array(starts, dtype=np.int64)
    except TypeError:  # None entries: no reference position for that call
        pos = np.array([-1 if p is None else p for p in starts], dtype=np.int64)
    keep = (pos >= start) & (pos < end) & (np.asarray(mods.ml) >= q_threshold)
    return pos[keep] - start


def get_m6a(fiber, start, end, ref, Q_THRESHOLD=200):
    context_length = end - start
    m6a_data = np.zeros(context_length, dtype=np.float32)

    overlap_start = max(start, fiber.start)
    overlap_end   = min(end,   fiber.end)
    if overlap_start >= overlap_end:
        return m6a_data

    lo, hi = overlap_start - start, overlap_end - start
    m6a_data[lo:hi][ref.at_mask[lo:hi]] = -1.0
    m6a_data[_call_positions(fiber.m6a, start, end, Q_THRESHOLD)] = 1.0
    return m6a_data


def get_cpg(fiber, start, end, ref, Q_THRESHOLD=200):
    context_length = end - start
    cpg_data = np.zeros(context_length, dtype=np.float32)

    overlap_start = max(start, fiber.start)
    overlap_end   = min(end,   fiber.end)
    if overlap_start < overlap_end:
        lo, hi = overlap_start - start, overlap_end - start
        cpg_data[lo:hi][ref.cpg_mask[lo:hi]] = -1.0

    cpg_data[_call_positions(fiber.cpg, start, end, Q_THRESHOLD)] = 1.0
    return cpg_data


def get_msp(fiber, start, end, ref, Q_THRESHOLD=0):
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


def get_nuc(fiber, start, end, ref, Q_THRESHOLD=0):
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


def get_fire_msp(fiber, start, end, ref, Q_THRESHOLD=200):
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

def nucleosome_span(fiber):
    """(first nucleosome start, last nucleosome end) in reference coordinates, or None if the fiber has no
    nucleosome with a reference position. Read ends outside this span are neither MSP nor nucleosome (MSPs are
    only called between nucleosomes), so their msp/nuc channels are uninformative there."""
    spans = [(s, s + n) for s, n in zip(fiber.nuc.reference_starts, fiber.nuc.reference_lengths)
             if s is not None and n is not None]
    if not spans:
        return None
    return min(s for s, _ in spans), max(e for _, e in spans)


def get_fiber_data(
    fiber_bam, chrom, start, end, fasta,
    fibers_per_entry, context_length, input_features,
    return_fiber_dna=False, min_overlap=50,
):
    """
    Collect fiber reads over [chrom:start-end] and extract per-fiber feature arrays.

    Each fiber has a scored span: its alignment trimmed to [first nucleosome start, last nucleosome end],
    intersected with the window. Input features cover the whole alignment (read ends still give context);
    the model only predicts, and coverage only counts, inside the scored span (fiber_mask).

    Parameters
    ----------
    fiber_bam       : open pyft.Fiberbam handle
    chrom, start, end : genomic locus
    fasta           : open pysam.FastaFile handle
    fibers_per_entry : max fibers to collect (N dimension)
    context_length  : window size in bp (= end - start)
    input_features  : list of signal functions [fn(fiber, start, end, ref: RefWindow) -> (L,)]
    return_fiber_dna : whether to build per-fiber DNA one-hot array
    min_overlap     : minimum bp of the fiber's scored span inside the window (fibers with less, or with no
                      nucleosome at all, are skipped)

    Returns
    -------
    fibers_array   : np.ndarray  (fibers_per_entry, n_features, context_length)
    dna_array      : np.ndarray  (fibers_per_entry, context_length, 4)  or  None
    n_fibers       : int  — actual fibers collected (≤ fibers_per_entry)
    fiber_coverage : np.ndarray  (context_length,)  — per-position count of fibers whose scored span covers it
    fiber_mask     : np.ndarray  (fibers_per_entry, context_length) bool — each fiber's scored span (padding rows False)
    """
    ref            = RefWindow(fasta.fetch(chrom, start, end))  # shared by every fiber in the window
    fibers_array   = np.zeros((fibers_per_entry, len(input_features), context_length), dtype=np.float32)
    dna_array      = np.zeros((fibers_per_entry, context_length, 4), dtype=np.float32) if return_fiber_dna else None
    fiber_mask     = np.zeros((fibers_per_entry, context_length), dtype=bool)

    with suppress_stdout_stderr():
        possible_fibers = fiber_bam.fetch(chrom, start, end)

    n_fibers = 0
    for fiber in possible_fibers:
        if n_fibers == fibers_per_entry:
            break

        span = nucleosome_span(fiber)
        if span is None:
            continue
        span_start = max(start, fiber.start, span[0])
        span_end   = min(end,   fiber.end,   span[1])
        if span_end - span_start < min_overlap:
            continue
        fiber_mask[n_fibers, span_start - start:span_end - start] = True

        overlap_start = max(start, fiber.start)
        overlap_end   = min(end,   fiber.end)
        win_start = overlap_start - start

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

        fibers_array[n_fibers] = np.array([fn(fiber, start, end, ref) for fn in input_features])
        n_fibers += 1

    fiber_coverage = fiber_mask.sum(axis=0).astype(np.float32)
    return fibers_array, dna_array, n_fibers, fiber_coverage, fiber_mask
