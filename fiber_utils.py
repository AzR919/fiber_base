

import numpy as np



def get_m6a(fiber, start, end, ref_dna_seq, Q_THRESHOLD=200):
    context_length = end-start
    m6a_data = np.zeros((context_length), dtype=np.float32)

    ref_seq_arr = np.array(list(ref_dna_seq.upper()))
    at_mask = np.isin(ref_seq_arr, ['A', 'T'])
    # Set all potential m6A target sites (A/T) to -1 (unmethylated background)
    m6a_data[at_mask] = -1.0

    ref_starts = np.array(fiber.m6a.reference_starts, dtype=np.float32)
    qualities = np.array(fiber.m6a.ml, dtype=np.float32)

    mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)
    valid_positions = (ref_starts[mask] - start).astype(np.int32)
    m6a_data[valid_positions] = 1
    return m6a_data


def get_cpg(fiber, start, end, ref_dna_seq, Q_THRESHOLD=200):
    context_length = end-start
    cpg_data = np.zeros(context_length, dtype=np.float32)
    ref_seq_arr = np.array(list(ref_dna_seq.upper()))
    L = len(ref_seq_arr)

    # 1. Identify CpG sites on the reference sequence
    # Forward strand: 'C' followed by 'G' -> mark the 'C' position
    c_positions = np.where(ref_seq_arr[:-1] == 'C')[0]
    valid_cg = c_positions[ref_seq_arr[c_positions + 1] == 'G']

    # Reverse strand: 'G' preceded by 'C' -> mark the 'G' position
    g_positions = np.where(ref_seq_arr[1:] == 'G')[0] + 1
    valid_gc = g_positions[ref_seq_arr[g_positions - 1] == 'C']

    # Combine all CpG positions and set them to -1.0 (unmethylated background)
    cpg_indices = np.unique(np.concatenate([valid_cg, valid_gc]))
    cpg_data[cpg_indices] = -1.0

    # 2. Extract methylated CpG positions from fiber
    ref_starts = np.array(fiber.cpg.reference_starts, dtype=np.float32)
    qualities = np.array(fiber.cpg.ml, dtype=np.float32)

    mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)
    valid_positions = (ref_starts[mask] - start).astype(np.int32)

    # 3. Mark identified methylated positions as 1.0
    cpg_data[valid_positions] = 1.0

    return cpg_data

def get_msp(fiber, start, end, ref_dna_seq, Q_THRESHOLD=0):
    context_length = end-start
    msp_data = np.zeros((context_length), dtype=np.float32)

    for ref_pos, length, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
        if ref_pos is None or length is None:
            continue

        ref_end = ref_pos + length
        if ref_pos < end and ref_end > start and aq >= Q_THRESHOLD:
            rel_start = ref_pos - start
            rel_end = ref_end - start

            win_start = max(0, rel_start)
            win_end = min(context_length, rel_end)
            msp_data[win_start:win_end] = 1

    return msp_data

def get_nuc(fiber, start, end, ref_dna_seq, Q_THRESHOLD=0):
    context_length = end-start
    nuc_data = np.zeros((context_length), dtype=np.float32)

    for ref_pos, length, aq in zip(fiber.nuc.reference_starts, fiber.nuc.reference_lengths, fiber.nuc.qual):
        if ref_pos is None or length is None:
            continue

        ref_end = ref_pos + length
        if ref_pos < end and ref_end > start and aq >= Q_THRESHOLD:
            rel_start = ref_pos - start
            rel_end = ref_end - start

            win_start = max(0, rel_start)
            win_end = min(context_length, rel_end)
            nuc_data[win_start:win_end] = 1

    return nuc_data

def get_fire_msp(fiber, start, end, ref_dna_seq, Q_THRESHOLD=200):
    context_length = end-start
    fire_msp_data = np.zeros((context_length), dtype=np.float32)

    # Access fire_msp if explicitly separated, else fallback to msp
    fire_source = getattr(fiber, 'fire_msp', fiber.msp)

    for ref_pos, length, aq in zip(fire_source.reference_starts, fire_source.reference_lengths, fire_source.qual):
        if ref_pos is None or length is None:
            continue

        ref_end = ref_pos + length
        if ref_pos < end and ref_end > start and aq >= Q_THRESHOLD:
            rel_start = ref_pos - start
            rel_end = ref_end - start

            win_start = max(0, rel_start)
            win_end = min(context_length, rel_end)
            fire_msp_data[win_start:win_end] = 1

    return fire_msp_data
"""
def get_fiber_data(fiber_bam, fibers_needed, input_flags, chrom, start, end, min_overlap=50):
    fibers_tensor = np.zeros((self.fibers_per_entry, len(self.input_features), self.context_length), dtype=np.float32)
    dna_tensor = np.zeros((self.fibers_per_entry, self.context_length, 4), dtype=np.float32) if self.return_fiber_dna else None

    ref_dna_seq = self.fasta.fetch(chrom, start, end)

    with suppress_stdout_stderr():
        possible_fibers = self.fiber_bams[cell_idx].fetch(chrom, start, end)

    i = 0
    for fiber in possible_fibers:
        if i == self.fibers_per_entry:
            break

        # Calculate overlap between read and target window
        overlap_start = max(start, fiber.start)
        overlap_end = min(end, fiber.end)
        overlap_len = overlap_end - overlap_start

        # Skip fibers that barely intersect the region
        if overlap_len < min_overlap:
            continue

        # Process optional per-fiber DNA sequence
        if self.return_fiber_dna:
            dna_buffer = list("N" * self.context_length)
            win_offset_start = overlap_start - start
            read_offset_start = overlap_start - fiber.start
            read_offset_end = overlap_end - fiber.start

            if fiber.seq is not None:
                read_seq_slice = fiber.seq[read_offset_start:read_offset_end]
                # Clamp slice length so slice substitution never alters array length
                slice_len = min(len(read_seq_slice), self.context_length - win_offset_start)
                if slice_len > 0:
                    dna_buffer[win_offset_start : win_offset_start + slice_len] = list(read_seq_slice[:slice_len])

            dna_tensor[i] = self.dna_to_onehot("".join(dna_buffer))

        # Feature functions handle boundary clipping safely
        single_fiber_data = np.array([func(fiber, start, end, ref_dna_seq) for func in self.input_features])
        fibers_tensor[i] = single_fiber_data
        i += 1

    fiber_dna_tensor = torch.from_numpy(dna_tensor).permute(2, 1, 0) if self.return_fiber_dna else None
    return torch.from_numpy(fibers_tensor).permute(1, 2, 0), fiber_dna_tensor, i
"""
