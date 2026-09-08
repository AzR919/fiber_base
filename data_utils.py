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

from pathlib import Path
from torch.utils.data import IterableDataset

from utils import *
from fiber_utils import (
    get_m6a, get_cpg, get_msp, get_nuc, get_fire_msp,
    get_fiber_data as _get_fiber_data_np,
    get_locus_onehot,
    suppress_stdout_stderr,
)

#--------------------------------------------------------------------------------------------------

class fiber_data_iterator(IterableDataset):

    def __init__(self, metadata, fibers_per_entry, context_length,
                 iters_per_epoch, input_flags, mode="train", seed=919,
                 dna_type="none", bulk_name="N/A"):

        self.metadata = metadata
        self.fasta_path = metadata["fasta_path"]
        self.bulk_name = bulk_name

        # Parse cell type mappings into lists of names and absolute file paths
        self.cell_type_names = list(metadata["cell_types"].keys())
        self.fiber_data_paths = []
        self.other_bw_paths = []

        fiber_base = metadata.get("fiber_base_path", "")
        bulk_base = metadata.get("bulk_base_path", "")

        for cell_name in self.cell_type_names:
            cram_file = metadata["cell_types"][cell_name]["fibers"]
            bw_file = metadata["cell_types"][cell_name]["bulk"]
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

        # DNA Return Options: "none", "ref", "fiber", "both"
        valid_dna_types = {"none", "ref", "fiber", "both"}
        if dna_type not in valid_dna_types:
            raise ValueError(f"Invalid dna_type '{dna_type}'. Expected one of {valid_dna_types}")

        self.dna_type = dna_type
        self.return_ref_genome = self.dna_type in ("ref", "both")
        self.return_fiber_dna = self.dna_type in ("fiber", "both")

        # Base random number generators
        self.rng = random.Random(seed)

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

    def onehot_for_locus(self, locus):
        """Fetch genomic DNA for a locus and return (context_length, 4) torch tensor."""
        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid genomic range: {start}-{end}")
        return torch.from_numpy(get_locus_onehot(self.fasta, chrom, start, end))

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
            self.fiber_bams = [pyft.Fiberbam(str(Path(p).resolve())) for p in self.fiber_data_paths]
        if self.other_bws is None:
            self.other_bws = [pyBigWig.open(p) for p in self.other_bw_paths]
        if self.fasta is None:
            if not os.path.exists(self.fasta_path):
                raise FileNotFoundError(f"FASTA not found: {self.fasta_path}")
            if not os.path.exists(self.fasta_path + ".fai"):
                pysam.faidx(self.fasta_path)
            self.fasta = pysam.FastaFile(self.fasta_path)

        feature_map = [get_m6a, get_cpg, get_msp, get_nuc, get_fire_msp]
        self.input_features = [feature_map[i] for i in self.active_feature_indices]

    def generate_random_locus(self):
        random_chr = self.rng.choice(list(self.chr_sizes.keys()))
        random_start = self.rng.randint(0, self.chr_sizes[random_chr] - self.context_length)
        random_end = random_start + self.context_length
        return random_chr, random_start, random_end

    def expand_ccre_locus(self, ccre_chrom, ccre_start, ccre_end, jitter_range=200):
        """Generates a genomic window centered around a random cCRE with optional jitter."""
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

    def generate_ccre_locus(self, jitter_range=200):
        """Generates a genomic window centered around a random cCRE with optional jitter."""
        ccre_chrom, ccre_start, ccre_end = self.rng.choice(self.ccre_list)
        return self.expand_ccre_locus(ccre_chrom, ccre_start, ccre_end, jitter_range)

    def _collect_fiber_tensors(self, cell_idx, chrom, start, end, min_overlap=50):
        """Thin torch wrapper around fiber_utils.get_fiber_data. Returns (C,L,N) tensors."""
        fibers_np, dna_np, n_fibers, coverage_np = _get_fiber_data_np(
            self.fiber_bams[cell_idx], chrom, start, end, self.fasta,
            self.fibers_per_entry, self.context_length, self.input_features,
            return_fiber_dna=self.return_fiber_dna, min_overlap=min_overlap,
        )
        fiber_dna_tensor = torch.from_numpy(dna_np).permute(2, 1, 0) if dna_np is not None else None
        return torch.from_numpy(fibers_np).permute(1, 2, 0), fiber_dna_tensor, n_fibers, torch.from_numpy(coverage_np)

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

        for _ in range(self.iters_per_epoch):
            found_possible_locus = False

            while not found_possible_locus:
                # Uniformly pick a cell type between all available samples
                cell_idx = self.rng.randint(0, self.num_cell_types - 1)
                cell_type_name = self.cell_type_names[cell_idx]

                random_locus = self.generate_ccre_locus()

                fiber_tensor, fiber_dna_tensor, n_fibers, fiber_coverage = self._collect_fiber_tensors(cell_idx, *random_locus, min_overlap=self.context_length//8)
                if n_fibers == 0:
                    continue

                other_tensor = self.get_other_bw_data(cell_idx, *random_locus)
                if torch.isnan(other_tensor).any().item():
                    continue

                ref_dna = self.onehot_for_locus(random_locus).T if self.return_ref_genome else None
                found_possible_locus = True

            out_dict = {
                "fiber_features": fiber_tensor,
                "target_bulk": other_tensor,
                "n_fibers": n_fibers,
                "fiber_coverage": fiber_coverage,
                "locus": random_locus,
                "cell_type": cell_type_name
            }

            if self.return_ref_genome:
                out_dict["ref_dna"] = ref_dna

            if self.return_fiber_dna:
                out_dict["fiber_dna_tensor"] = fiber_dna_tensor

            yield out_dict


#--------------------------------------------------------------------------------------------------
# Testing

def tester():
    config_path = "./configs/evals/eval00.yaml"
    with open(config_path, "r") as f:
        eval_config = yaml.safe_load(f)

    kwargs = {
        "metadata": eval_config["metadata"],
        "fibers_per_entry": 100,
        "context_length": 20,
        "iters_per_epoch": 50,
        "input_flags": [1, 1, 0, 0, 0],
        "mode": "train",
        "dna_type": "both"
    }

    t_set = fiber_data_iterator(**kwargs)

    # Single-instance check using dictionary unpacking
    sample_dict = next(iter(t_set))
    print(f"Single instance check -> Cell: {sample_dict['cell_type']}, Locus: {sample_dict['locus']}, Fibers: {sample_dict['n_fibers']}")
    print(f"Keys present in dictionary (dna_type='{kwargs['dna_type']}'): {list(sample_dict.keys())}")

    # Loop inspection
    for i, batch in enumerate(t_set):
        print(f"Sample {i+1} | Locus: {batch['locus']} | Cell: {batch['cell_type']} | Fibers: {batch['n_fibers']} | Bulk Shape: {batch['target_bulk'].shape}")

    print("All done")

if __name__ == "__main__":
    tester()
