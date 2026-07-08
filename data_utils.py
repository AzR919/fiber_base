"""
Data processing file

"""

import os
import pyft
import pysam
import torch
import random
import pyBigWig
import numpy as np
import pandas as pd

# from w_redirect import stdout_redirected
from torch.utils.data import IterableDataset

from utils import *

class fiber_data_iterator(IterableDataset):

    def __init__(self, fiber_data_path, other_bw,
                 fibers_per_entry, context_length,
                 iters_per_epoch, fasta_path,
                 input_flags, ccre_path,
                 chr_sizes_file=None, mode="train",
                 seed=919):

        # Store paths instead of opening the file pointers globally
        self.fiber_data_path = fiber_data_path
        self.other_bw_path = other_bw
        self.fasta_path = fasta_path

        self.fiber_bam = None
        self.other_bw = None
        self.fasta = None

        self.fibers_per_entry = fibers_per_entry
        self.context_length = context_length
        self.iters_per_epoch = iters_per_epoch
        self.seed = seed
        self.epoch = 0  # Added tracking to allow fresh shuffling per training epoch
        self.mode = mode

        # Initialize base generators for parsing setup files (like cCREs)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Temporary initialization of BigWig just to extract metadata cleanly
        temp_bw = pyBigWig.open(other_bw)
        self.load_genomic_coords(temp_bw.chroms(), mode=mode)
        temp_bw.close()

        self.load_ccres(ccre_path, mode=mode)

        self.input_flags = input_flags

        # We store the indices to map features inside the workers safely
        self.active_feature_indices = [i for i in range(5) if self.input_flags[i]]

    def set_epoch(self, epoch):
        """Call this at the beginning of your training loop: train_dataset.set_epoch(epoch)"""
        self.epoch = epoch

    def load_fasta(self, fasta_path):

        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")
        if not os.path.exists(fasta_path + ".fai"):
            pysam.faidx(fasta_path)               # build index if needed
        self.fasta = pysam.FastaFile(fasta_path)

    def dna_to_onehot(self, sequence):
            # Create a mapping from nucleotide to index
            mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}

            # Convert the sequence to indices
            indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)

            # Create one-hot encoding
            one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

            # Remove the fifth column which corresponds to 'N'
            one_hot = one_hot[:, :4]

            return one_hot.to(torch.float32)

    def onehot_for_locus(self, locus):
        """
        Helper to fetch DNA and convert to one-hot for a given locus [chrom, start, end].
        Returns a tensor [context_length, 4].

        """
        def get_DNA_sequence(chrom, start, end):
            """
            Retrieve the sequence for a given chromosome and coordinate range from a fasta file.

            """
            # Ensure coordinates are within the valid range
            if start < 0 or end <= start:
                raise ValueError("Invalid start or end position")

            return self.fasta.fetch(chrom, start, end)

        def dna_to_onehot(sequence):
            # Create a mapping from nucleotide to index
            mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}

            # Convert the sequence to indices
            indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)

            # Create one-hot encoding
            one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

            # Remove the fifth column which corresponds to 'N'
            one_hot = one_hot[:, :4]

            return one_hot.to(torch.float32)

        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        seq = get_DNA_sequence(chrom, start, end)
        return dna_to_onehot(seq)

    def load_genomic_coords(self, possible_chr_sizes, mode="train"):
        if mode == "train":
            main_chrs = ["chr20"]
        elif "val" in mode:
            main_chrs = ["chr21"]
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.chr_sizes = {k: possible_chr_sizes[k] for k in main_chrs if k in possible_chr_sizes}

    def load_ccres(self, bed_path, mode="train"):
        df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2])
        df.columns = ['chrom', 'start', 'end']
        filtered_df = df[df['chrom'].isin(self.chr_sizes.keys())]

        if mode == "val10":
            filtered_df = filtered_df.sample(frac=0.10, random_state=self.np_rng)

        self.ccre_list = filtered_df.values

    def init_worker_resources(self):
        """Safely instantiates file descriptors unique to the background worker thread."""
        if self.fiber_bam is None:
            self.fiber_bam = pyft.Fiberbam(self.fiber_data_path)
        if self.other_bw is None:
            self.other_bw = pyBigWig.open(self.other_bw_path)
        if self.fasta is None:
            if not os.path.exists(self.fasta_path):
                raise FileNotFoundError(f"FASTA not found: {self.fasta_path}")
            if not os.path.exists(self.fasta_path + ".fai"):
                pysam.faidx(self.fasta_path)
            self.fasta = pysam.FastaFile(self.fasta_path)

        # Map methods to child instances safely
        feature_map = [
            self.get_m6a,
            self.get_cpg,
            self.get_msp,
            self.get_nuc,
            self.get_fire_msp,
        ]
        self.input_features = [feature_map[i] for i in self.active_feature_indices]

    def generate_loci(self):

        random_chr = self.rng.choice(list(self.chr_sizes.keys()))

        random_start = self.rng.randint(0, self.chr_sizes[random_chr])
        random_end = random_start + self.context_length

        return random_chr, random_start, random_end

    def generate_ccre_loci(self, jitter_range=200):
        """
        Generates a genomic window centered around a random cCRE with optional jitter.

        @args:
            jitter_range (int): The maximum number of base pairs to shift the center.
                                e.g., 200 means a shift between -200 and +200 bp.
        """
        # 1. Pick a random cCRE
        ccre_chrom, ccre_start, ccre_end = self.rng.choice(self.ccre_list)

        # 2. Calculate the "true" center of the cCRE
        true_center = (ccre_start + ccre_end) // 2

        # 3. Apply Jitter
        # This shifts the focus point slightly so the cCRE isn't always perfectly centered
        jitter = self.rng.randint(-jitter_range, jitter_range)
        focal_point = true_center + jitter

        # 4. Create the window around the focal point
        half_window = self.context_length // 2
        random_start = focal_point - half_window
        random_end = random_start + self.context_length

        # 5. Boundary Check (Crucial to prevent index errors)
        max_size = self.chr_sizes[ccre_chrom]

        if random_start < 0:
            random_start = 0
            random_end = self.context_length
        elif random_end > max_size:
            random_end = max_size
            random_start = max_size - self.context_length

        return ccre_chrom, int(random_start), int(random_end)

    def get_m6a(self, fiber, start, end, Q_THRESHOLD=200):

        m6a_data = np.zeros((self.context_length), dtype=np.float32)

        # for ref_pos, aq in zip(fiber.m6a.reference_starts, fiber.m6a.ml):
        #     if ref_pos is None: continue
        #     if start <= ref_pos < end and aq >= Q_THRESHOLD:
        #         m6a_data[ref_pos-start:ref_pos-start+len] = 1

        # 1. Convert lists to numpy arrays
        ref_starts = np.array(fiber.m6a.reference_starts, dtype=np.float32)
        qualities = np.array(fiber.m6a.ml, dtype=np.float32)

        # 2. Create a boolean mask for everything that passes the filters
        # - Within the genomic window
        # - Above the quality threshold
        # - Not None (numpy handles this well if converted correctly)
        mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)

        # 3. Extract the passing positions and calculate their relative offsets
        valid_positions = (ref_starts[mask] - start).astype(np.int32)

        # 4. Use "Fancy Indexing" to set all 1s at once
        m6a_data[valid_positions] = 1

        return m6a_data

    def get_cpg(self, fiber, start, end, Q_THRESHOLD=200):

        cpg_data = np.zeros((self.context_length), dtype=np.float32)

        # 1. Convert lists to numpy arrays
        ref_starts = np.array(fiber.cpg.reference_starts, dtype=np.float32)
        qualities = np.array(fiber.cpg.ml, dtype=np.float32)

        # 2. Create a boolean mask for everything that passes the filters
        # - Within the genomic window
        # - Above the quality threshold
        # - Not None (numpy handles this well if converted correctly)
        mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)

        # 3. Extract the passing positions and calculate their relative offsets
        valid_positions = (ref_starts[mask] - start).astype(np.int32)

        # 4. Use "Fancy Indexing" to set all 1s at once
        cpg_data[valid_positions] = 1

        return cpg_data

    def get_msp(self, fiber, start, end, Q_THRESHOLD=0):

        msp_data = np.zeros((self.context_length), dtype=np.float32)

        for ref_pos, len, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
            if ref_pos is None: continue
            if start <= ref_pos < end and aq >= Q_THRESHOLD:
                msp_data[ref_pos-start:ref_pos-start+len] = 1

        return msp_data

    def get_nuc(self, fiber, start, end, Q_THRESHOLD=0):

        nuc_data = np.zeros((self.context_length), dtype=np.float32)

        for ref_pos, len, aq in zip(fiber.nuc.reference_starts, fiber.nuc.reference_lengths, fiber.nuc.qual):
            if ref_pos is None: continue
            if start <= ref_pos < end and aq >= Q_THRESHOLD:
                nuc_data[ref_pos-start:ref_pos-start+len] = 1

        return nuc_data

    def get_fire_msp(self, fiber, start, end, Q_THRESHOLD=200):

        fire_msp_data = np.zeros((self.context_length), dtype=np.float32)

        for ref_pos, len, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
            if ref_pos is None: continue
            if start <= ref_pos < end and aq >= Q_THRESHOLD:
                fire_msp_data[ref_pos-start:ref_pos-start+len] = 1

        return fire_msp_data

    def _extract_single_fiber(self, fiber, start, end):
        dna_fiber = fiber.seq[start - fiber.start:start - fiber.start + self.context_length]
        if len(dna_fiber) != self.context_length:
            return None

        single_fiber_data = np.array([func(fiber, start, end) for func in self.input_features])
        dna_onehot = self.dna_to_onehot(dna_fiber)
        return single_fiber_data, dna_onehot

    def _fetch_fibers_from_bam(self, fiber_bam, chrom, start, end):
        collected = []
        with suppress_stdout_stderr():
            possible_fibers = fiber_bam.fetch(chrom, start, end)

        for fiber in possible_fibers:
            extracted = self._extract_single_fiber(fiber, start, end)
            if extracted is not None:
                collected.append(extracted)

        return collected

    def _pack_fiber_batch(self, fiber_samples):
        fibers_tensor = np.zeros(
            (self.fibers_per_entry, len(self.input_features), self.context_length),
            dtype=np.float32,
        )
        dna_tensor = np.zeros(
            (self.fibers_per_entry, self.context_length, 4),
            dtype=np.float32,
        )

        for i, (single_fiber_data, dna_onehot) in enumerate(fiber_samples):
            fibers_tensor[i] = single_fiber_data
            dna_tensor[i] = dna_onehot

        return (
            torch.from_numpy(fibers_tensor).permute(1, 2, 0),
            torch.from_numpy(dna_tensor).permute(2, 1, 0),
        )

    def get_fiber_data(self, chrom, start, end):
        fiber_samples = self._fetch_fibers_from_bam(self.fiber_bam, chrom, start, end)
        if len(fiber_samples) < self.fibers_per_entry:
            return None, None

        return self._pack_fiber_batch(fiber_samples[:self.fibers_per_entry])

    def get_other_bw_data(self, chrom, start, end):

        return None, None, torch.asinh(torch.from_numpy(np.array(self.other_bw.values(chrom, start, end))).to(torch.float32))

    def __iter__(self):
        # 1. Initialize file descriptors inside the worker loop
        self.init_worker_resources()

        # 2. Add self.epoch into the worker seed calculation to rotate sequences per epoch
        worker_info = torch.utils.data.get_worker_info()

        if "val" in self.mode:
            seed_offset = 0
        elif worker_info is None:
            seed_offset = self.epoch * 1000
        else:
            seed_offset = worker_info.id + self.epoch * 1000

        worker_seed = self.seed + seed_offset

        self.rng = random.Random(worker_seed)
        self.np_rng = np.random.default_rng(worker_seed)

        for _ in range(self.iters_per_epoch):
            found_possible_locus = False

            while not found_possible_locus:
                random_locus = self.generate_ccre_loci()

                fiber_tensor, dna_tensor = self.get_fiber_data(*random_locus)
                if fiber_tensor is None: continue

                tensor_a, tensor_b, other_tensor = self.get_other_bw_data(*random_locus)
                has_nan = torch.isnan(other_tensor).any().item()
                if has_nan: continue

                dna = self.onehot_for_locus(random_locus)
                found_possible_locus = True

            yield fiber_tensor, dna, dna_tensor, other_tensor, random_locus, tensor_a, tensor_b


class mixed_cell_fiber_data_iterator(fiber_data_iterator):
    """
    Mixed-cell iterator: blends two Fiber-seq inputs and two BigWig targets.

    For mix_fraction alpha in [0, 1]:
      - target = alpha * bw_a + (1 - alpha) * bw_b
      - input  = random subsample of fibers_per_entry with
                 round(alpha * N) from cell A and the rest from cell B
    """

    def __init__(
        self,
        fiber_data_path_a,
        fiber_data_path_b,
        other_bw_a,
        other_bw_b,
        mix_fraction,
        fibers_per_entry,
        context_length,
        iters_per_epoch,
        fasta_path,
        input_flags,
        ccre_path,
        chr_sizes_file=None,
        mode="train",
        seed=919,
    ):
        if not 0.0 <= mix_fraction <= 1.0:
            raise ValueError(f"mix_fraction must be in [0, 1], got {mix_fraction}")

        self.fiber_data_path_b = fiber_data_path_b
        self.other_bw_path_b = other_bw_b
        self.mix_fraction = mix_fraction

        super().__init__(
            fiber_data_path_a,
            other_bw_a,
            fibers_per_entry=fibers_per_entry,
            context_length=context_length,
            iters_per_epoch=iters_per_epoch,
            fasta_path=fasta_path,
            input_flags=input_flags,
            ccre_path=ccre_path,
            chr_sizes_file=chr_sizes_file,
            mode=mode,
            seed=seed,
        )

        self.fiber_bam_b = None
        self.other_bw_b = None

    def _fibers_per_cell(self):
        n_a = int(round(self.mix_fraction * self.fibers_per_entry))
        n_b = self.fibers_per_entry - n_a
        return n_a, n_b

    def init_worker_resources(self):
        super().init_worker_resources()
        if self.fiber_bam_b is None:
            self.fiber_bam_b = pyft.Fiberbam(self.fiber_data_path_b)
        if self.other_bw_b is None:
            self.other_bw_b = pyBigWig.open(self.other_bw_path_b)

    def get_fiber_data(self, chrom, start, end):
        n_a, n_b = self._fibers_per_cell()

        fibers_a = self._fetch_fibers_from_bam(self.fiber_bam, chrom, start, end)
        fibers_b = self._fetch_fibers_from_bam(self.fiber_bam_b, chrom, start, end)

        if len(fibers_a) < n_a or len(fibers_b) < n_b:
            return None, None

        sampled = self.rng.sample(fibers_a, n_a) + self.rng.sample(fibers_b, n_b)
        self.rng.shuffle(sampled)
        return self._pack_fiber_batch(sampled)

    def arc_sined(self, np_arr):
        return torch.asinh(torch.from_numpy(np_arr).to(torch.float32))

    def get_other_bw_data(self, chrom, start, end):
        vals_a = np.array(self.other_bw.values(chrom, start, end), dtype=np.float32)
        vals_b = np.array(self.other_bw_b.values(chrom, start, end), dtype=np.float32)
        blended = self.mix_fraction * vals_a + (1.0 - self.mix_fraction) * vals_b
        return self.arc_sined(vals_a), self.arc_sined(vals_b), self.arc_sined(blended)

def make_fiber_data_iterator(
    fiber_data_path,
    other_data_path,
    fibers_per_entry,
    context_length,
    iters_per_epoch,
    fasta_path,
    input_flags,
    ccre_path,
    mode="train",
    seed=919,
    fiber_data_path_b=None,
    other_data_path_b=None,
    mix_fraction=1.0,
):
    common_kwargs = dict(
        fibers_per_entry=fibers_per_entry,
        context_length=context_length,
        iters_per_epoch=iters_per_epoch,
        fasta_path=fasta_path,
        input_flags=input_flags,
        ccre_path=ccre_path,
        mode=mode,
        seed=seed,
    )

    if fiber_data_path_b is not None or other_data_path_b is not None:
        if fiber_data_path_b is None or other_data_path_b is None:
            raise ValueError(
                "Mixed-cell mode requires both --fiber_data_path_b and --other_data_path_b"
            )
        return mixed_cell_fiber_data_iterator(
            fiber_data_path,
            fiber_data_path_b,
            other_data_path,
            other_data_path_b,
            mix_fraction=mix_fraction,
            **common_kwargs,
        )

    return fiber_data_iterator(fiber_data_path, other_data_path, **common_kwargs)


#--------------------------------------------------------------------------------------------------
# testing

def tester():

    kwargs = {
        "fiber_data_path":"/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/GM12878-fire-v0.1-filtered.cram",
        "fibers_per_entry": 200,
        "context_length": 20,
        "iters_per_epoch": 1000,
        "fasta_path": "/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa",
        "input_flags": [1,1,1,1,1],
        "ccre_path": "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/gm12878_ccres.bed"
    }
    other_bw = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/ENCFF603BJO_ATAC_seq.bigWig"

    t_set = fiber_data_iterator(other_bw=other_bw, **kwargs)

    sample = next(iter(t_set))

    fiber_path_b = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/K562/K562-fire-v0.1-filtered.cram"
    bw_path_b = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/K562/ENCFF071GML_H3K4me3_signal.bigWig"
    mix_frac = 0.5

    t_set_m = make_fiber_data_iterator(other_data_path=other_bw, other_data_path_b=bw_path_b, fiber_data_path_b=fiber_path_b, mix_fraction=mix_frac, **kwargs)

    sample_m = next(iter(t_set_m))
    i=0
    for _ in t_set_m:
        print(i)
        i+=1
        pass

    pass

if __name__=="__main__":

    tester()
