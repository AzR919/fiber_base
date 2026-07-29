"""
Data processing file for multi-cell type fiber and bulk chromatin accessibility model.
"""

import os
import pyft
import pysam
import torch
import random
import pyBigWig
import numpy as np
import pandas as pd

from torch.utils.data import IterableDataset

from utils import *

#--------------------------------------------------------------------------------------------------

class fiber_data_iterator(IterableDataset):

    def __init__(self, metadata, fibers_per_entry, context_length,
                 iters_per_epoch, input_flags, mode="train", seed=919,
                 return_dna=False):

        self.metadata = metadata
        self.fasta_path = metadata["fasta_path"]

        # Parse cell type mappings into lists of names and absolute file paths
        self.cell_type_names = list(metadata["cell_types"].keys())
        self.fiber_data_paths = []
        self.other_bw_paths = []

        fiber_base = metadata.get("fiber_base_path", "")
        bulk_base = metadata.get("bulk_base_path", "")

        for cell_name in self.cell_type_names:
            cram_file, bw_file = metadata["cell_types"][cell_name]
            self.fiber_data_paths.append(os.path.join(fiber_base, cram_file))
            self.other_bw_paths.append(os.path.join(bulk_base, bw_file))

        self.num_cell_types = len(self.cell_type_names)

        # File handles (instantiated per worker process in init_worker_resources)
        self.fiber_bams = None
        self.other_bws = None
        self.fasta = None

        self.fibers_per_entry = fibers_per_entry
        self.context_length = context_length
        self.iters_per_epoch = iters_per_epoch
        self.seed = seed
        self.epoch = 0
        self.mode = mode
        self.return_dna = return_dna

        # Base random number generators
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Read chromosome sizes directly from the FASTA index (.fai)
        if not os.path.exists(self.fasta_path + ".fai"):
            pysam.faidx(self.fasta_path)

        fasta_idx = pysam.FastaFile(self.fasta_path)
        chrom_sizes = dict(zip(fasta_idx.references, fasta_idx.lengths))
        fasta_idx.close()

        self.load_genomic_sizes(chrom_sizes, mode=mode)
        self.load_ccres(metadata["ccre_path"], mode=mode)

        self.input_flags = input_flags
        self.active_feature_indices = [i for i in range(5) if self.input_flags[i]]

    def set_epoch(self, epoch):
        """Call this at the beginning of your training loop: train_dataset.set_epoch(epoch)"""
        self.epoch = epoch

    def dna_to_onehot(self, sequence):
        """
        Convert a nucleotide sequence string into a 4-channel one-hot PyTorch tensor.
        'N's map to channel index 4, which is removed, producing zero vectors [0,0,0,0] for padding.
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        indices = torch.tensor([mapping.get(nuc.upper(), 4) for nuc in sequence], dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=5)
        # Remove fifth column ('N' representation / zero-padding)
        return one_hot[:, :4].to(torch.float32)

    def onehot_for_locus(self, locus):
        """Helper to fetch genomic DNA sequence for a locus [chrom, start, end] and return [context_length, 4]."""
        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid genomic range: {start}-{end}")
        seq = self.fasta.fetch(chrom, start, end)
        return self.dna_to_onehot(seq)

    def load_genomic_sizes(self, possible_chr_sizes, mode="train"):
        """Filter chromosomes based on train/val chromosomes specified in metadata."""
        if mode == "train":
            target_chrs = self.metadata.get("train_chrs", ["chr20"])
        elif mode == "val":
            target_chrs = self.metadata.get("val_chrs", ["chr21"])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.chr_sizes = {k: possible_chr_sizes[k] for k in target_chrs if k in possible_chr_sizes}
        if not self.chr_sizes:
            raise ValueError(f"None of the requested target chromosomes {target_chrs} were found in the FASTA index!")

    def load_ccres(self, bed_path, mode="train"):
        df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2])
        df.columns = ['chrom', 'start', 'end']
        filtered_df = df[df['chrom'].isin(self.chr_sizes.keys())]

        self.ccre_list = filtered_df.values

    def init_worker_resources(self):
        """Safely instantiates file descriptors unique to each background worker process/thread."""
        if self.fiber_bams is None:
            self.fiber_bams = [pyft.Fiberbam(p) for p in self.fiber_data_paths]
        if self.other_bws is None:
            self.other_bws = [pyBigWig.open(p) for p in self.other_bw_paths]
        if self.fasta is None:
            if not os.path.exists(self.fasta_path):
                raise FileNotFoundError(f"FASTA not found: {self.fasta_path}")
            if not os.path.exists(self.fasta_path + ".fai"):
                pysam.faidx(self.fasta_path)
            self.fasta = pysam.FastaFile(self.fasta_path)

        feature_map = [
            self.get_m6a,
            self.get_cpg,
            self.get_msp,
            self.get_nuc,
            self.get_fire_msp,
        ]
        self.input_features = [feature_map[i] for i in self.active_feature_indices]

    def generate_random_locus(self):
        random_chr = self.rng.choice(list(self.chr_sizes.keys()))
        random_start = self.rng.randint(0, self.chr_sizes[random_chr] - self.context_length)
        random_end = random_start + self.context_length
        return random_chr, random_start, random_end

    def generate_ccre_locus(self, jitter_range=200):
        """Generates a genomic window centered around a random cCRE with optional jitter."""
        ccre_chrom, ccre_start, ccre_end = self.rng.choice(self.ccre_list)
        true_center = (ccre_start + ccre_end) // 2

        jitter = self.rng.randint(-jitter_range, jitter_range)
        focal_point = true_center + jitter

        half_window = self.context_length // 2
        random_start = focal_point - half_window
        random_end = random_start + self.context_length

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
        ref_starts = np.array(fiber.m6a.reference_starts, dtype=np.float32)
        qualities = np.array(fiber.m6a.ml, dtype=np.float32)

        mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)
        valid_positions = (ref_starts[mask] - start).astype(np.int32)
        m6a_data[valid_positions] = 1
        return m6a_data

    def get_cpg(self, fiber, start, end, Q_THRESHOLD=200):
        cpg_data = np.zeros((self.context_length), dtype=np.float32)
        ref_starts = np.array(fiber.cpg.reference_starts, dtype=np.float32)
        qualities = np.array(fiber.cpg.ml, dtype=np.float32)

        mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)
        valid_positions = (ref_starts[mask] - start).astype(np.int32)
        cpg_data[valid_positions] = 1
        return cpg_data

    def get_msp(self, fiber, start, end, Q_THRESHOLD=0):
        msp_data = np.zeros((self.context_length), dtype=np.float32)

        for ref_pos, length, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
            if ref_pos is None or length is None:
                continue

            ref_end = ref_pos + length
            if ref_pos < end and ref_end > start and aq >= Q_THRESHOLD:
                rel_start = ref_pos - start
                rel_end = ref_end - start

                win_start = max(0, rel_start)
                win_end = min(self.context_length, rel_end)
                msp_data[win_start:win_end] = 1

        return msp_data

    def get_nuc(self, fiber, start, end, Q_THRESHOLD=0):
        nuc_data = np.zeros((self.context_length), dtype=np.float32)

        for ref_pos, length, aq in zip(fiber.nuc.reference_starts, fiber.nuc.reference_lengths, fiber.nuc.qual):
            if ref_pos is None or length is None:
                continue

            ref_end = ref_pos + length
            if ref_pos < end and ref_end > start and aq >= Q_THRESHOLD:
                rel_start = ref_pos - start
                rel_end = ref_end - start

                win_start = max(0, rel_start)
                win_end = min(self.context_length, rel_end)
                nuc_data[win_start:win_end] = 1

        return nuc_data

    def get_fire_msp(self, fiber, start, end, Q_THRESHOLD=200):
        fire_msp_data = np.zeros((self.context_length), dtype=np.float32)

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
                win_end = min(self.context_length, rel_end)
                fire_msp_data[win_start:win_end] = 1

        return fire_msp_data

    def get_fiber_data(self, cell_idx, chrom, start, end, min_overlap=50):
        fibers_tensor = np.zeros((self.fibers_per_entry, len(self.input_features), self.context_length), dtype=np.float32)
        dna_tensor = np.zeros((self.fibers_per_entry, self.context_length, 4), dtype=np.float32) if self.return_dna else None

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
            if self.return_dna:
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
            single_fiber_data = np.array([func(fiber, start, end) for func in self.input_features])
            fibers_tensor[i] = single_fiber_data
            i += 1

        fiber_dna_out = torch.from_numpy(dna_tensor).permute(2, 1, 0) if self.return_dna else None
        return torch.from_numpy(fibers_tensor).permute(1, 2, 0), fiber_dna_out, i

    def get_other_bw_data(self, cell_idx, chrom, start, end):
        raw_vals = np.array(self.other_bws[cell_idx].values(chrom, start, end), dtype=np.float32)
        # Handle potential missing/NaN coverage regions in BigWig
        raw_vals = np.nan_to_num(raw_vals, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.asinh(torch.from_numpy(raw_vals))

    def __iter__(self):
        self.init_worker_resources()

        worker_info = torch.utils.data.get_worker_info()
        if self.mode == "val": # consistent val set across epochs
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
                # Uniformly pick a cell type between all available samples
                cell_idx = self.rng.randint(0, self.num_cell_types - 1)
                cell_type_name = self.cell_type_names[cell_idx]

                random_locus = self.generate_ccre_locus()

                fiber_tensor, fiber_dna_tensor, n_fibers = self.get_fiber_data(cell_idx, *random_locus, min_overlap=self.context_length//8)
                if n_fibers == 0:
                    continue

                other_tensor = self.get_other_bw_data(cell_idx, *random_locus)
                if torch.isnan(other_tensor).any().item():
                    continue

                genomic_dna_tensor = self.onehot_for_locus(random_locus) if self.return_dna else None
                found_possible_locus = True

            out_dict = {
                "fiber_features": fiber_tensor,
                "target_bulk": other_tensor,
                "num_fibers": n_fibers,
                "locus": random_locus,
                "cell_type": cell_type_name
            }

            if self.return_dna:
                out_dict["genomic_dna"] = genomic_dna_tensor
                out_dict["fiber_dna"] = fiber_dna_tensor

            yield out_dict


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    data_root = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER"

    metadata = {
        "fasta_path": "/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa",
        "ccre_path": "/home/azr/projects/def-maxwl/azr/data/misc/grch38_ccres.bed",
        "train_chrs": ["chr20"],
        "val_chrs": ["chr21"],
        "fiber_base_path": f"{data_root}/fiber_multi_cell",
        "bulk_base_path": f"{data_root}/atac_multi_cell",
        # Map cell type names directly to their (CRAM, BigWig) tuple
        "cell_types": {
            "GM12878": (
                "GM12878-fire-v0.1-filtered.cram",
                "GM12878_ENCFF603BJO_ATAC_seq_fcc.bigWig"
            ),
            "K562": (
                "K562_Fiber_seq_200U_1M_cells_200U_PS01370-fire-v0.1-filtered.cram",
                "K562_ENCFF102ARJ_ATAC_seq_fcc.bigWig"
            )
        }
    }

    kwargs = {
        "metadata": metadata,
        "fibers_per_entry": 200,
        "context_length": 4096,
        "iters_per_epoch": 5,
        "input_flags": [1, 1, 1, 1, 1],
        "mode": "train",
        "return_dna": True
    }

    t_set = fiber_data_iterator(**kwargs)

    # Single-instance check using dictionary unpacking
    sample_dict = next(iter(t_set))
    print(f"Single instance check -> Cell: {sample_dict['cell_type']}, Locus: {sample_dict['locus']}, Fibers: {sample_dict['num_fibers']}")
    print(f"Keys present in dictionary (return_dna={kwargs['return_dna']}): {list(sample_dict.keys())}")

    # Loop inspection
    for i, batch in enumerate(t_set):
        print(f"Sample {i+1} | Locus: {batch['locus']} | Cell: {batch['cell_type']} | Fibers: {batch['num_fibers']} | Bulk Shape: {batch['target_bulk'].shape}")

    print("All done")

if __name__ == "__main__":
    tester()
