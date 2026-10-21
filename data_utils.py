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
                  With baseline_cells ({cell: metadata entry}, the cells a model was trained on), samples also
                  carry the cross-cell-average baseline (see get_baseline).
    With DataLoader workers, training samples and eval regions are split across workers.
    """

    def __init__(self, metadata, fibers_per_entry, context_length,
                 mode, input_flags, seed=919,
                 dna_type="none", output_assays=None,
                 samples_per_epoch=1000, train_windows=None, windows=None, max_regions=None,
                 baseline_cells=None, baseline_include_own=False):

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

        # Cross-cell-average baseline: one source per training cell line (samples sharing an ENCODE folder, like
        # K562_100U and K562_200U, count once); each evaluated cell averages every source except its own cell line,
        # or every source with baseline_include_own (the mixture reference of T3: the training mixture's track).
        self.baseline_sources, self.baseline_bws = None, None
        if baseline_cells:
            if mode != "eval":
                raise ValueError("baseline_cells is only supported in eval mode.")
            by_line = {}
            for entry in baseline_cells.values():
                by_line.setdefault(entry["encode_dir"], [entry["bulk"].get(a) for a in self.output_assays])
            self.baseline_sources = list(by_line.items())                    # [(encode_dir, [path or None per assay])]
            own = [metadata["cell_types"][c].get("encode_dir") for c in self.cell_type_names]
            self.baseline_index = [[i for i, (d, _) in enumerate(self.baseline_sources)
                                    if baseline_include_own or d != own_dir]
                                   for own_dir in own]

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
        if self.baseline_sources and self.baseline_bws is None:
            self.baseline_bws = [[None if p is None else pyBigWig.open(p) for p in paths]
                                 for _, paths in self.baseline_sources]
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
        """Fibers at a locus as a dict: fiber_features (C, L, N), fiber_dna_tensor (4, L, N) or None, n_fibers,
        fiber_mask (L, N) bool (each fiber's scored span, see fiber_utils.get_fiber_data) and fiber_coverage (L,)."""
        fibers_np, dna_np, n_fibers, coverage_np, mask_np = _get_fiber_data_np(
            self.fiber_bams[cell_idx], chrom, start, end, self.fasta,
            self.fibers_per_entry, self.context_length, self.input_features,
            return_fiber_dna=self.return_fiber_dna, min_overlap=min_overlap,
        )
        return {
            "fiber_features": torch.from_numpy(fibers_np).permute(1, 2, 0),
            "fiber_dna_tensor": torch.from_numpy(dna_np).permute(2, 1, 0) if dna_np is not None else None,
            "n_fibers": n_fibers,
            "fiber_mask": torch.from_numpy(mask_np).T.contiguous(),
            "fiber_coverage": torch.from_numpy(coverage_np),
        }

    @staticmethod
    def _read_track(bw, chrom, start, end):
        """(L,) raw bigWig values; NaN (no data) -> 0, so targets never contain NaN and samples need no NaN check."""
        raw = np.asarray(bw.values(chrom, start, end, numpy=True), dtype=np.float32)
        return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    def get_other_bw_data(self, cell_idx, chrom, start, end):
        """(K, L) asinh bulk signal; assays this cell has no data for are zeros (see self.assay_masks)."""
        tracks = [np.zeros(end - start, dtype=np.float32) if bw is None else self._read_track(bw, chrom, start, end)
                  for bw in self.other_bws[cell_idx]]
        return torch.asinh(torch.from_numpy(np.stack(tracks, axis=0)))  # (K, L)

    def get_baseline(self, cell_idx, chrom, start, end):
        """
        Cross-cell-average baseline for an evaluated cell: per assay, the mean over the training cell lines (all
        except the evaluated cell's own line, unless baseline_include_own) of their fold change at this window,
        then asinh, i.e. averaged where
        tracks are linear (as a pooled population of cells) and reported in the same space as the targets.
        Returns (bulk (K, L), mask (K,)); mask 0 where no training cell line has the assay.
        """
        K, L = len(self.output_assays), end - start
        total, count = np.zeros((K, L), dtype=np.float64), np.zeros(K, dtype=np.float64)
        for i in self.baseline_index[cell_idx]:
            for k, bw in enumerate(self.baseline_bws[i]):
                if bw is not None:
                    total[k] += self._read_track(bw, chrom, start, end)
                    count[k] += 1
        mean_fc = total / np.maximum(count, 1)[:, None]
        return (torch.from_numpy(np.arcsinh(mean_fc).astype(np.float32)),
                torch.from_numpy((count > 0).astype(np.float32)))

    def strata_for(self, locus):
        return torch.from_numpy(self.regions.strata_masks(*locus))

    def _make_sample(self, fibers, other_tensor, locus, cell_type_name, assay_mask, strata=None, baseline=None):
        out_dict = {
            "fiber_features": fibers["fiber_features"],
            "target_bulk": other_tensor,
            "assay_mask": assay_mask,
            "n_fibers": fibers["n_fibers"],
            "fiber_mask": fibers["fiber_mask"],
            "fiber_coverage": fibers["fiber_coverage"],
            "locus": locus,
            "cell_type": cell_type_name
        }
        if strata is not None:
            out_dict["strata"] = strata
        if baseline is not None:
            out_dict["baseline_bulk"], out_dict["baseline_mask"] = baseline
        if self.return_ref_genome:
            out_dict["ref_dna"] = self.onehot_for_locus(locus).T
        if self.return_fiber_dna:
            out_dict["fiber_dna_tensor"] = fibers["fiber_dna_tensor"]
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

                    fibers = self._collect_fiber_tensors(cell_idx, *locus, min_overlap=self.context_length // 8)
                    if fibers["n_fibers"] == 0:
                        continue

                    other_tensor = self.get_other_bw_data(cell_idx, *locus)
                    found = True

                yield self._make_sample(fibers, other_tensor, locus, cell_type_name, self.assay_masks[cell_idx])

        else:  # eval mode: deterministic, no retry
            for region in self._eval_regions():
                locus = self.window_for_region(region)
                strata = self.strata_for(locus)

                for cell_idx, cell_type_name in enumerate(self.cell_type_names):
                    fibers = self._collect_fiber_tensors(cell_idx, *locus, min_overlap=self.context_length // 8)
                    if fibers["n_fibers"] == 0:
                        continue

                    other_tensor = self.get_other_bw_data(cell_idx, *locus)
                    baseline = self.get_baseline(cell_idx, *locus) if self.baseline_sources else None
                    yield self._make_sample(fibers, other_tensor, locus, cell_type_name, self.assay_masks[cell_idx],
                                            strata=strata, baseline=baseline)


#--------------------------------------------------------------------------------------------------

class MixedCellFiberDataset(SingleCellFiberDataset):
    """
    Composite samples for mixture training (T3 in reports/TODO.md): fibers from several cell types at the same
    window, with the bulk target that mixture would have.

    Proportions: with mix_range [lo, hi] (exactly 2 cell types), the first cell type's share of the fibers is drawn
    uniformly from [lo, hi] for every sample (eval mode: seeded per region, so validation is reproducible);
    otherwise fixed from each cell type's `ratio` in the config (uniform if absent).
    Fibers: each cell type contributes a random subset of its fibers at the window, round(share × fibers_per_entry),
    or all it has if fewer (the sample is then smaller, not skipped). Windows where a cell type has no fiber are
    retried (train) or skipped (eval).
    Target: per position x, the fold-change mixture of the cell types' tracks weighted by the fibers actually
    present there, asinh(sum_c q_c(x) sinh(t_c(x))) with q_c(x) = cell c's fibers covering x / all fibers covering
    x. This is exactly what the model's fiber average (fiber_aggregation fc) can match if every fiber carried its
    cell type's value; the nominal share only decides how fibers are sampled (spans differ along the window).
    Extra keys: fiber_origin (N,) int64 (index into cell_type_names, -1 = padding), mix_fraction (n_cells,) the
    realized share of fibers, cell_targets (n_cells, K, L) each cell type's own track.
    """

    def __init__(self, metadata, fibers_per_entry, context_length,
                 input_flags, mode='eval', seed=919,
                 dna_type="none", output_assays=None,
                 samples_per_epoch=1000, train_windows=None, windows=None, max_regions=None, mix_range=None):

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

        if mix_range is not None:
            lo, hi = (float(v) for v in mix_range)
            if self.num_cell_types != 2 or not 0 <= lo <= hi <= 1:
                raise ValueError(f"mix_range {mix_range} needs exactly 2 cell types and 0 <= lo <= hi <= 1 "
                                 f"(cell types: {self.cell_type_names}).")
            self.mix_range = (lo, hi)
        else:
            self.mix_range = None
            ratios = [(metadata["cell_types"][ct] or {}).get("ratio", 1.0) for ct in self.cell_type_names]
            self.fixed_shares = [r / sum(ratios) for r in ratios]

    @property
    def expected_samples(self):
        """Eval mode: one composite per region if none is skipped."""
        return len(self.region_list)

    def _draw_shares(self, rng):
        if self.mix_range is None:
            return self.fixed_shares
        p = rng.uniform(*self.mix_range)
        return [p, 1.0 - p]

    def _sample(self, rng, locus):
        """One composite at a window, or None if some cell type has no fiber there."""
        N = self.fibers_per_entry
        shares = self._draw_shares(rng)
        wanted = [int(round(s * N)) for s in shares[:-1]]
        wanted.append(N - sum(wanted))

        picked = []
        for cell_idx, n_want in enumerate(wanted):
            fibers = self._collect_fiber_tensors(cell_idx, *locus, min_overlap=self.context_length // 8)
            if fibers["n_fibers"] == 0:
                return None
            take = min(n_want, fibers["n_fibers"])
            idx = sorted(rng.sample(range(fibers["n_fibers"]), take))   # random subset, not the first by start
            picked.append((cell_idx, fibers, torch.tensor(idx, dtype=torch.long)))

        C, L, K = len(self.input_features), self.context_length, len(self.output_assays)
        features = torch.zeros(C, L, N)
        mask = torch.zeros(L, N, dtype=torch.bool)
        origin = torch.full((N,), -1, dtype=torch.long)
        dna = torch.zeros(4, L, N) if self.return_fiber_dna else None
        cell_targets = torch.zeros(self.num_cell_types, K, L)
        offset = 0
        for cell_idx, fibers, idx in picked:
            n = len(idx)
            if n:
                features[:, :, offset:offset + n] = fibers["fiber_features"][:, :, idx]
                mask[:, offset:offset + n] = fibers["fiber_mask"][:, idx]
                if dna is not None:
                    dna[:, :, offset:offset + n] = fibers["fiber_dna_tensor"][:, :, idx]
            origin[offset:offset + n] = cell_idx
            cell_targets[cell_idx] = self.get_other_bw_data(cell_idx, *locus)
            offset += n

        # Per-position share of each cell type among the fibers present -> fold-change mixture of their tracks
        cov = torch.stack([mask[:, origin == c].sum(-1) for c in range(self.num_cell_types)]).double()   # (n_cells, L)
        q = cov / cov.sum(0).clamp(min=1)
        target = torch.asinh((q[:, None, :] * torch.sinh(cell_targets.double())).sum(0)).float()      # (K, L)

        out = {
            "fiber_features": features,
            "target_bulk": target,
            # A mixed target is only valid for assays every contributing cell type has
            "assay_mask": torch.stack([self.assay_masks[c] for c in range(self.num_cell_types)]).prod(dim=0),
            "n_fibers": offset,
            "fiber_mask": mask,
            "fiber_coverage": mask.sum(-1).float(),
            "fiber_origin": origin,
            "mix_fraction": torch.tensor([(origin == c).sum().item() / offset for c in range(self.num_cell_types)]),
            "cell_targets": cell_targets,
            "locus": locus,
            "cell_type": "mixed",
        }
        if self.return_ref_genome:
            out["ref_dna"] = self.onehot_for_locus(locus).T
        if dna is not None:
            out["fiber_dna_tensor"] = dna
        return out

    def __iter__(self):
        self.init_worker_resources()

        if self.mode == "train":
            for _ in range(self._train_samples()):
                sample = None
                while sample is None:
                    sample = self._sample(self.rng, self.window_at(*self.sampler.sample_centre(self.rng)))
                yield sample

        else:  # eval mode: deterministic (proportions and fiber subsets seeded per region), no retry
            for region in self._eval_regions():
                locus = self.window_for_region(region)
                sample = self._sample(random.Random(f"{self.seed}:{region[0]}:{region[1]}"), locus)
                if sample is None:
                    continue
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
