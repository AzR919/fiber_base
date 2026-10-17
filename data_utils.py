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
from regions import Regions, TrainSampler, DEFAULT_TRAIN_WINDOWS
from fiber_utils import (
    get_m6a, get_cpg, get_msp, get_nuc, get_fire_msp,
    get_fiber_data as _get_fiber_data_np,
    get_locus_onehot,
    suppress_stdout_stderr,
)

#--------------------------------------------------------------------------------------------------

def loader_context(num_workers):
    """multiprocessing_context for DataLoaders. Workers must not be plain forks of a process that has already
    read fibers: pyft's Rust thread pool does not survive fork() and the workers hang silently. 'forkserver'
    forks them from a clean helper process instead (scripts using it need an `if __name__ == "__main__":` guard)."""
    return "forkserver" if num_workers > 0 else None


def _worker_shard():
    """(worker id, number of workers) for the current DataLoader worker; (0, 1) without workers."""
    info = torch.utils.data.get_worker_info()
    return (0, 1) if info is None else (info.id, info.num_workers)


class SingleCellFiberDataset(IterableDataset):
    """
    Iterable dataset yielding one (fiber stack, bulk target) pair per cell type per region.
    Regions come from the region files (see regions.py / configs/README.md); the input window is
    context_length bp centred on each region.

    mode='train': draws region centres from the training pools (train_windows: sources + jitter) on
                  train_chrs, retries regions where the cell has no fibers, seeded per epoch and worker.
    mode='eval':  iterates a fixed window set (windows: e.g. val_chr22_n200, test_chr21_tiles) × all
                  cell types, no retry (regions without fibers are skipped), fully reproducible. Samples
                  also carry 'strata': (len(STRATA), L) per-bp cCRE-class masks for stratified metrics.
    With DataLoader workers, training samples and eval regions are split across workers.
    """

    def __init__(self, metadata, fibers_per_entry, context_length,
                 mode, input_flags, seed=919,
                 dna_type="none", output_assays=None,
                 samples_per_epoch=1000, train_windows=None, windows=None, max_regions=None):

        if output_assays is None:
            output_assays = ["atac"]
        self.output_assays = output_assays

        self.metadata = metadata
        self.fasta_path = metadata["fasta_path"]

        self.cell_type_names = list(metadata["cell_types"].keys())
        self.fiber_data_paths = []
        self.other_bw_paths = []  # list of lists: [cell_idx][assay_idx] -> path

        fiber_base = metadata.get("fiber_base_path", "")
        bulk_bases = metadata.get("bulk_base_paths", {})

        for cell_name in self.cell_type_names:
            cell = metadata["cell_types"][cell_name]
            self.fiber_data_paths.append(os.path.join(fiber_base, cell["fibers"]))
            # None = this cell has no bulk data for that assay (masked out of the loss)
            bw_paths = [None if cell["bulk"].get(a) is None else os.path.join(bulk_bases.get(a, ""), cell["bulk"][a])
                        for a in self.output_assays]
            self.other_bw_paths.append(bw_paths)

        # Per cell type: (K,) float mask, 1 where the assay has bulk data
        self.assay_masks = [torch.tensor([p is not None for p in paths], dtype=torch.float32)
                            for paths in self.other_bw_paths]

        self.num_cell_types = len(self.cell_type_names)

        self.fiber_bams = None
        self.other_bws = None
        self.fasta = None

        self.fibers_per_entry = fibers_per_entry
        self.context_length = context_length
        self.samples_per_epoch = samples_per_epoch
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

        self.regions = Regions(metadata["regions_root"])
        if context_length > self.regions.max_window:
            raise ValueError(f"context_length {context_length} exceeds the regions' max_window "
                             f"{self.regions.max_window}; raise max_window in data_prep/regions_spec.yaml.")
        if context_length != self.regions.region_length:
            print(f"[regions] context_length {context_length} != region_length {self.regions.region_length}: "
                  f"windows are centred on regions and scored over the whole window (centre-only scoring "
                  f"is not implemented yet).", flush=True)

        if mode == "train":
            tw = train_windows or DEFAULT_TRAIN_WINDOWS
            self.sampler = TrainSampler(self.regions, metadata["train_chrs"], tw["sources"], tw.get("jitter", 0))
        elif mode == "eval":
            if not windows:
                raise ValueError("mode='eval' needs a window set name (windows=...), e.g. 'test_chr21_mini2000'.")
            self.windows = windows
            self.region_list = self.regions.window_set(windows, max_regions=max_regions, seed=seed)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'train' or 'eval'.")

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

    @property
    def expected_samples(self):
        """Eval mode: samples yielded if no region is skipped (regions × cell types)."""
        return len(self.region_list) * self.num_cell_types

    def init_worker_resources(self):
        if self.fiber_bams is None:
            self.fiber_bams = [pyft.Fiberbam(str(Path(p).resolve())) for p in self.fiber_data_paths]
        if self.other_bws is None:
            self.other_bws = [[None if p is None else pyBigWig.open(p) for p in paths] for paths in self.other_bw_paths]
        if self.fasta is None:
            if not os.path.exists(self.fasta_path):
                raise FileNotFoundError(f"FASTA not found: {self.fasta_path}")
            if not os.path.exists(self.fasta_path + ".fai"):
                pysam.faidx(self.fasta_path)
            self.fasta = pysam.FastaFile(self.fasta_path)

        feature_map = [get_m6a, get_cpg, get_msp, get_nuc, get_fire_msp]
        self.input_features = [feature_map[i] for i in self.active_feature_indices]

    def window_at(self, chrom, centre):
        """Input window (chrom, start, end) of context_length bp centred on a region centre. The region rule
        guarantees it lies inside the chromosome, so no clamping is needed."""
        start = int(centre) - self.context_length // 2
        return chrom, start, start + self.context_length

    def window_for_region(self, region):
        chrom, start, end = region
        return self.window_at(chrom, (start + end) // 2)

    def _train_samples(self):
        """Seeds self.rng for this epoch and worker; returns this worker's share of samples_per_epoch."""
        wid, nw = _worker_shard()
        self.rng = random.Random(self.seed + wid + self.epoch * 1000)
        return self.samples_per_epoch // nw + (1 if wid < self.samples_per_epoch % nw else 0)

    def _eval_regions(self):
        """This worker's share of the window set (strided, so each region is evaluated exactly once)."""
        wid, nw = _worker_shard()
        return self.region_list[wid::nw]

    def _collect_fiber_tensors(self, cell_idx, chrom, start, end, min_overlap=50):
        fibers_np, dna_np, n_fibers, coverage_np = _get_fiber_data_np(
            self.fiber_bams[cell_idx], chrom, start, end, self.fasta,
            self.fibers_per_entry, self.context_length, self.input_features,
            return_fiber_dna=self.return_fiber_dna, min_overlap=min_overlap,
        )
        fiber_dna_tensor = torch.from_numpy(dna_np).permute(2, 1, 0) if dna_np is not None else None
        return torch.from_numpy(fibers_np).permute(1, 2, 0), fiber_dna_tensor, n_fibers, torch.from_numpy(coverage_np)

    def get_other_bw_data(self, cell_idx, chrom, start, end):
        """(K, L) asinh bulk signal; assays this cell has no data for are zeros (see self.assay_masks)."""
        tracks = []
        for bw in self.other_bws[cell_idx]:
            if bw is None:
                tracks.append(np.zeros(end - start, dtype=np.float32))
                continue
            raw = np.array(bw.values(chrom, start, end), dtype=np.float32)
            tracks.append(np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0))
        arr = np.stack(tracks, axis=0)  # (K, L)
        return torch.asinh(torch.from_numpy(arr))

    def strata_for(self, locus):
        return torch.from_numpy(self.regions.strata_masks(*locus))

    def _make_sample(self, fiber_tensor, fiber_dna_tensor, n_fibers, fiber_coverage,
                     other_tensor, locus, cell_type_name, assay_mask, strata=None):
        out_dict = {
            "fiber_features": fiber_tensor,
            "target_bulk": other_tensor,
            "assay_mask": assay_mask,
            "n_fibers": n_fibers,
            "fiber_coverage": fiber_coverage,
            "locus": locus,
            "cell_type": cell_type_name
        }
        if strata is not None:
            out_dict["strata"] = strata
        if self.return_ref_genome:
            out_dict["ref_dna"] = self.onehot_for_locus(locus).T
        if self.return_fiber_dna:
            out_dict["fiber_dna_tensor"] = fiber_dna_tensor
        return out_dict

    def __iter__(self):
        self.init_worker_resources()

        if self.mode == "train":
            for _ in range(self._train_samples()):
                found = False
                while not found:
                    cell_idx = self.rng.randint(0, self.num_cell_types - 1)
                    cell_type_name = self.cell_type_names[cell_idx]
                    locus = self.window_at(*self.sampler.sample_centre(self.rng))

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
                                        other_tensor, locus, cell_type_name, self.assay_masks[cell_idx])

        else:  # eval mode: deterministic, no retry
            for region in self._eval_regions():
                locus = self.window_for_region(region)
                strata = self.strata_for(locus)

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
                                            other_tensor, locus, cell_type_name, self.assay_masks[cell_idx],
                                            strata=strata)


#--------------------------------------------------------------------------------------------------

class MixedCellFiberDataset(SingleCellFiberDataset):
    """
    Dataset that samples fibers from multiple cell types at a shared locus, concatenating
    them into a synthetic mixed stack with per-cell-type masks and bulk targets.

    mode='train': draws regions from the training pools (as SingleCellFiberDataset), retries if any
                  cell type fails.
    mode='eval':  iterates a fixed window set, one composite per region, skipping regions where any
                  required cell type has insufficient fibers.
    """

    def __init__(self, metadata, fibers_per_entry, context_length,
                 input_flags, mode='eval', seed=919,
                 dna_type="none", output_assays=None,
                 samples_per_epoch=1000, train_windows=None, windows=None, max_regions=None):

        super().__init__(
            metadata=metadata,
            fibers_per_entry=fibers_per_entry,
            context_length=context_length,
            samples_per_epoch=samples_per_epoch,
            input_flags=input_flags,
            mode=mode,
            seed=seed,
            dna_type=dna_type,
            output_assays=output_assays,
            train_windows=train_windows,
            windows=windows,
            max_regions=max_regions,
        )

        try:
            meta_cell_ratios = {ct: metadata["cell_types"][ct]["ratio"] for ct in metadata["cell_types"].keys()}
        except Exception:
            print("Failed to extract cell ratio from meta. Falling back to uniform distribution")
            meta_cell_ratios = None

        self.cell_ratios = self._setup_mixing_ratios(meta_cell_ratios)
        self.fiber_counts_per_cell = self._calculate_fiber_counts()

    @property
    def expected_samples(self):
        """Eval mode: one composite per region if none is skipped."""
        return len(self.region_list)

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
            "assay_mask": self.assay_masks[cell_idx],
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
            # A mixed target is only valid for assays every contributing cell type has
            "assay_mask": torch.stack([s["assay_mask"] for s in cell_samples]).prod(dim=0),
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
            for _ in range(self._train_samples()):
                found = False
                while not found:
                    locus = self.window_at(*self.sampler.sample_centre(self.rng))

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
            for region in self._eval_regions():
                locus = self.window_for_region(region)
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

                sample = self._build_composite_sample(locus, cell_samples)
                sample["strata"] = self.strata_for(locus)
                yield sample


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
