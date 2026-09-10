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

class SingleCellFiberDataset(IterableDataset):
    """
    Iterable dataset yielding one (fiber stack, bulk target) pair per cell type per locus.

    mode='train': randomly samples cCREs with jitter, retries on failure, seeded per epoch.
    mode='eval':  deterministically iterates ccre_list[:num_sample_ccres] × all cell types,
                  no jitter, skips loci that fail (no retry), fully reproducible.
    """

    def __init__(self, metadata, fibers_per_entry, context_length,
                 mode, input_flags, seed=919,
                 dna_type="none", bulk_name="N/A",
                 iters_per_epoch=1000, num_sample_ccres=100):

        self.metadata = metadata
        self.fasta_path = metadata["fasta_path"]
        self.bulk_name = bulk_name

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

        self.fiber_bams = None
        self.other_bws = None
        self.fasta = None

        self.fibers_per_entry = fibers_per_entry
        self.context_length = context_length
        self.iters_per_epoch = iters_per_epoch
        self.num_sample_ccres = num_sample_ccres
        self.seed = seed
        self.epoch = 0
        self.mode = mode

        valid_dna_types = {"none", "ref", "fiber", "both"}
        if dna_type not in valid_dna_types:
            raise ValueError(f"Invalid dna_type '{dna_type}'. Expected one of {valid_dna_types}")

        self.dna_type = dna_type
        self.return_ref_genome = self.dna_type in ("ref", "both")
        self.return_fiber_dna = self.dna_type in ("fiber", "both")

        self.rng = random.Random(seed)

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
        """Call at the start of each training epoch for seeded-deterministic data ordering."""
        self.epoch = epoch

    def onehot_for_locus(self, locus):
        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid genomic range: {start}-{end}")
        return torch.from_numpy(get_locus_onehot(self.fasta, chrom, start, end))

    def load_genomic_sizes(self, possible_chr_sizes, mode="train"):
        if mode == "train":
            target_chrs = self.metadata.get("train_chrs", ["chr20"])
        elif mode == "eval":
            target_chrs = self.metadata.get("val_chrs", ["chr21"])
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'train' or 'eval'.")

        self.chr_sizes = {k: possible_chr_sizes[k] for k in target_chrs if k in possible_chr_sizes}
        if not self.chr_sizes:
            raise ValueError(f"None of the requested chromosomes {target_chrs} were found in the FASTA index!")

    def load_ccres(self, bed_path, mode="train"):
        df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2])
        df.columns = ['chrom', 'start', 'end']
        filtered_df = df[df['chrom'].isin(self.chr_sizes.keys())]
        self.ccre_list = filtered_df.values

    def init_worker_resources(self):
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

    def expand_ccre_locus(self, ccre_chrom, ccre_start, ccre_end, jitter_range=200):
        true_center = (ccre_start + ccre_end) // 2
        jitter = self.rng.randint(-jitter_range, jitter_range) if jitter_range > 0 else 0
        focal_point = true_center + jitter

        half_window = self.context_length // 2
        start = focal_point - half_window
        end = start + self.context_length

        max_size = self.chr_sizes[ccre_chrom]
        if start < 0:
            start = 0
            end = self.context_length
        elif end > max_size:
            end = max_size
            start = max_size - self.context_length

        return ccre_chrom, int(start), int(end)

    def generate_ccre_locus(self, jitter_range=200):
        ccre_chrom, ccre_start, ccre_end = self.rng.choice(self.ccre_list)
        return self.expand_ccre_locus(ccre_chrom, ccre_start, ccre_end, jitter_range)

    def _collect_fiber_tensors(self, cell_idx, chrom, start, end, min_overlap=50):
        fibers_np, dna_np, n_fibers, coverage_np = _get_fiber_data_np(
            self.fiber_bams[cell_idx], chrom, start, end, self.fasta,
            self.fibers_per_entry, self.context_length, self.input_features,
            return_fiber_dna=self.return_fiber_dna, min_overlap=min_overlap,
        )
        fiber_dna_tensor = torch.from_numpy(dna_np).permute(2, 1, 0) if dna_np is not None else None
        return torch.from_numpy(fibers_np).permute(1, 2, 0), fiber_dna_tensor, n_fibers, torch.from_numpy(coverage_np)

    def get_other_bw_data(self, cell_idx, chrom, start, end):
        raw_vals = np.array(self.other_bws[cell_idx].values(chrom, start, end), dtype=np.float32)
        raw_vals = np.nan_to_num(raw_vals, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.asinh(torch.from_numpy(raw_vals))

    def _make_sample(self, fiber_tensor, fiber_dna_tensor, n_fibers, fiber_coverage,
                     other_tensor, locus, cell_type_name):
        out_dict = {
            "fiber_features": fiber_tensor,
            "target_bulk": other_tensor,
            "n_fibers": n_fibers,
            "fiber_coverage": fiber_coverage,
            "locus": locus,
            "cell_type": cell_type_name
        }
        if self.return_ref_genome:
            out_dict["ref_dna"] = self.onehot_for_locus(locus).T
        if self.return_fiber_dna:
            out_dict["fiber_dna_tensor"] = fiber_dna_tensor
        return out_dict

    def __iter__(self):
        self.init_worker_resources()

        if self.mode == "train":
            worker_info = torch.utils.data.get_worker_info()
            seed_offset = 0 if worker_info is None else worker_info.id
            seed_offset += self.epoch * 1000
            self.rng = random.Random(self.seed + seed_offset)

            for _ in range(self.iters_per_epoch):
                found = False
                while not found:
                    cell_idx = self.rng.randint(0, self.num_cell_types - 1)
                    cell_type_name = self.cell_type_names[cell_idx]
                    locus = self.generate_ccre_locus()

                    fiber_tensor, fiber_dna_tensor, n_fibers, fiber_coverage = self._collect_fiber_tensors(
                        cell_idx, *locus, min_overlap=self.context_length // 8
                    )
                    if n_fibers == 0:
                        continue

                    other_tensor = self.get_other_bw_data(cell_idx, *locus)
                    if torch.isnan(other_tensor).any().item():
                        continue

                    found = True

                yield self._make_sample(fiber_tensor, fiber_dna_tensor, n_fibers, fiber_coverage,
                                        other_tensor, locus, cell_type_name)

        else:  # eval mode: deterministic, no retry
            ccre_iter = self.ccre_list if self.num_sample_ccres == -1 else self.ccre_list[:self.num_sample_ccres]

            for ccre_entry in ccre_iter:
                locus = self.expand_ccre_locus(*ccre_entry, jitter_range=0)

                for cell_idx, cell_type_name in enumerate(self.cell_type_names):
                    fiber_tensor, fiber_dna_tensor, n_fibers, fiber_coverage = self._collect_fiber_tensors(
                        cell_idx, *locus, min_overlap=self.context_length // 8
                    )
                    if n_fibers == 0:
                        continue

                    other_tensor = self.get_other_bw_data(cell_idx, *locus)
                    if torch.isnan(other_tensor).any().item():
                        continue

                    yield self._make_sample(fiber_tensor, fiber_dna_tensor, n_fibers, fiber_coverage,
                                            other_tensor, locus, cell_type_name)


#--------------------------------------------------------------------------------------------------

class MixedCellFiberDataset(SingleCellFiberDataset):
    """
    Dataset that samples fibers from multiple cell types at a shared locus, concatenating
    them into a synthetic mixed stack with per-cell-type masks and bulk targets.

    mode='train': randomly samples cCREs with jitter, retries if any cell type fails.
    mode='eval':  deterministically iterates ccre_list[:num_sample_ccres], no jitter,
                  skips loci where any required cell type has insufficient fibers.
    """

    def __init__(self, metadata, fibers_per_entry, context_length,
                 input_flags, mode='eval', seed=919,
                 dna_type="none", bulk_name="N/A",
                 iters_per_epoch=1000, num_sample_ccres=100):

        super().__init__(
            metadata=metadata,
            fibers_per_entry=fibers_per_entry,
            context_length=context_length,
            iters_per_epoch=iters_per_epoch,
            num_sample_ccres=num_sample_ccres,
            input_flags=input_flags,
            mode=mode,
            seed=seed,
            dna_type=dna_type,
            bulk_name=bulk_name,
        )

        try:
            meta_cell_ratios = {ct: metadata["cell_types"][ct]["ratio"] for ct in metadata["cell_types"].keys()}
        except Exception:
            print("Failed to extract cell ratio from meta. Falling back to uniform distribution")
            meta_cell_ratios = None

        self.cell_ratios = self._setup_mixing_ratios(meta_cell_ratios)
        self.fiber_counts_per_cell = self._calculate_fiber_counts()

    def _setup_mixing_ratios(self, cell_ratios):
        if cell_ratios is None:
            uniform_p = 1.0 / self.num_cell_types
            return {ct: uniform_p for ct in self.cell_type_names}
        total_p = sum(cell_ratios.values())
        return {ct: cell_ratios[ct] / total_p for ct in self.cell_type_names}

    def _calculate_fiber_counts(self):
        counts = {}
        allocated = 0
        sorted_cts = sorted(self.cell_type_names)
        for i, ct in enumerate(sorted_cts):
            if i == len(sorted_cts) - 1:
                counts[ct] = self.fibers_per_entry - allocated
            else:
                c = int(round(self.cell_ratios[ct] * self.fibers_per_entry))
                counts[ct] = c
                allocated += c
        return counts

    def _sample_single_cell_type(self, cell_idx, ct, locus):
        n_fibers_needed = self.fiber_counts_per_cell[ct]
        if n_fibers_needed == 0:
            return None

        fiber_tensor, fiber_dna_tensor, n_fibers, fiber_coverage = self._collect_fiber_tensors(
            cell_idx, *locus, min_overlap=self.context_length // 8
        )

        if n_fibers < n_fibers_needed:
            return None

        bw_tensor = self.get_other_bw_data(cell_idx, *locus)
        if torch.isnan(bw_tensor).any().item():
            return None

        trimmed_fiber_tensor = fiber_tensor[:, :, :n_fibers_needed]
        trimmed_dna_tensor = (
            fiber_dna_tensor[:, :, :n_fibers_needed]
            if (self.return_fiber_dna and fiber_dna_tensor is not None)
            else None
        )

        return {
            "cell_type": ct,
            "fiber_features": trimmed_fiber_tensor,
            "target_bulk": bw_tensor,
            "fiber_dna_tensor": trimmed_dna_tensor,
            "n_fibers": n_fibers_needed,
            "fiber_coverage": fiber_coverage,
        }

    def _build_composite_sample(self, locus, cell_samples):
        mixed_fiber_tensors = np.zeros(
            (len(self.input_features), self.context_length, self.fibers_per_entry), dtype=np.float32
        )
        mixed_dna_tensors = (
            np.zeros((4, self.context_length, self.fibers_per_entry), dtype=np.float32)
            if self.return_fiber_dna else None
        )

        cell_type_masks = {}
        individual_bulk_targets = {}
        mixed_coverage = torch.zeros(self.context_length, dtype=torch.float32)
        current_fiber_offset = 0

        for sample in cell_samples:
            ct = sample["cell_type"]
            n_fibers = sample["n_fibers"]

            individual_bulk_targets[ct] = sample["target_bulk"]
            mixed_fiber_tensors[:, :, current_fiber_offset:current_fiber_offset + n_fibers] = (
                sample["fiber_features"][:, :, :n_fibers]
            )
            mixed_coverage += sample["fiber_coverage"]

            if self.return_fiber_dna and sample["fiber_dna_tensor"] is not None:
                mixed_dna_tensors[:, :, current_fiber_offset:current_fiber_offset + n_fibers] = (
                    sample["fiber_dna_tensor"][:, :, :n_fibers]
                )

            mask = torch.zeros(self.fibers_per_entry, dtype=torch.bool)
            mask[current_fiber_offset:current_fiber_offset + n_fibers] = True
            cell_type_masks[ct] = mask
            current_fiber_offset += n_fibers

        composite_bulk = torch.stack(
            [self.cell_ratios[ct] * individual_bulk_targets[ct] for ct in individual_bulk_targets],
            dim=0
        ).sum(dim=0)

        out_dict = {
            "fiber_features": torch.from_numpy(mixed_fiber_tensors),
            "target_bulk": composite_bulk,
            "cell_type": "mixed",
            "cell_type_targets": individual_bulk_targets,
            "cell_type_masks": cell_type_masks,
            "n_fibers": current_fiber_offset,
            "fiber_coverage": mixed_coverage,
            "locus": locus,
            "mixing_ratios": self.cell_ratios,
        }

        if self.return_ref_genome:
            out_dict["ref_dna"] = self.onehot_for_locus(locus).T

        if self.return_fiber_dna:
            out_dict["fiber_dna_tensor"] = torch.from_numpy(mixed_dna_tensors)

        return out_dict

    def __iter__(self):
        self.init_worker_resources()

        if self.mode == "train":
            worker_info = torch.utils.data.get_worker_info()
            seed_offset = 0 if worker_info is None else worker_info.id
            seed_offset += self.epoch * 1000
            self.rng = random.Random(self.seed + seed_offset)

            for _ in range(self.iters_per_epoch):
                found = False
                while not found:
                    ccre_entry = self.rng.choice(self.ccre_list)
                    locus = self.expand_ccre_locus(*ccre_entry, jitter_range=200)

                    cell_samples = []
                    failed = False
                    for cell_idx, ct in enumerate(self.cell_type_names):
                        sample = self._sample_single_cell_type(cell_idx, ct, locus)
                        if sample is None and self.fiber_counts_per_cell[ct] > 0:
                            failed = True
                            break
                        if sample is not None:
                            cell_samples.append(sample)

                    if not failed and cell_samples:
                        found = True

                yield self._build_composite_sample(locus, cell_samples)

        else:  # eval mode: deterministic, no retry
            ccre_iter_list = self.ccre_list if self.num_sample_ccres == -1 else self.ccre_list[:self.num_sample_ccres]

            for ccre_locus in ccre_iter_list:
                locus = self.expand_ccre_locus(*ccre_locus, jitter_range=0)
                cell_samples = []
                failed_sampling = False

                for cell_idx, ct in enumerate(self.cell_type_names):
                    sample = self._sample_single_cell_type(cell_idx, ct, locus)
                    if sample is None and self.fiber_counts_per_cell[ct] > 0:
                        failed_sampling = True
                        break
                    if sample is not None:
                        cell_samples.append(sample)

                if failed_sampling or not cell_samples:
                    continue

                yield self._build_composite_sample(locus, cell_samples)


#--------------------------------------------------------------------------------------------------

def make_fiber_dataset(dataset_type: str, mode: str, **kwargs) -> IterableDataset:
    """
    Factory function for dataset construction.

    dataset_type: 'single' | 'mixed'
    mode:         'train'  | 'eval'
    """
    if dataset_type == "single":
        return SingleCellFiberDataset(mode=mode, **kwargs)
    elif dataset_type == "mixed":
        return MixedCellFiberDataset(mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}. Expected 'single' or 'mixed'.")
