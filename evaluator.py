"""
Evaluator module for testing fiber data models.
Takes a model and a dataset, runs inference, and computes MSE / Pearson metrics per cell type (an eval config can
list several; mixed composites count as cell type "mixed").

Optionally scores the cross-cell-average baseline (the mean of the training cells' tracks, see
data_utils.SingleCellFiberDataset.get_baseline) on exactly the same samples and positions, and the cell-specific
correlation: the partial correlation of prediction and target given the baseline (how much of the target the
prediction explains beyond what the cross-cell average already does). With model=None only the baseline is scored.
With baseline_include_own the baseline is the mixture reference of T3 (the training cells' mixture, own line
included), and the partial correlation is the cell-specific r given the mixture.

Loci breakdown (eval config `loci`): metrics restricted to the tiles of cell-specific / pairwise loci sets
(regions.Loci), per assay. Paired readouts (eval config `paired: true`, exactly 2 cell types A, B): both cells are
evaluated at every window, and the evaluator also reports
  - differential correlation corr(p_A - p_B, t_A - t_B) per region type / loci set, at positions scored in both;
  - sign accuracy at the differential bins of `pair_loci` (does p_A - p_B have the sign of d = t_A - t_B?);
  - fiber-origin AUC at those bins: per fiber its mean prediction over the bin, AUC of A's fibers vs B's
    (oriented by the sign of d, so 0.5 = no separation, 1 = A's fibers perfectly ranked on the right side).
See reports/TODO.md, T3.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data_utils import make_fiber_dataset, loader_context
from metrics import masked_mse, per_assay_mse, PearsonMeter, score_weights, partial_correlation, MIN_COVERAGE
from regions import STRATA, Loci
from utils import *
from vis_utils import plot_evaluator_record_t, plot_paired_locus, plot_differential_scatter, plot_fiber_auc

# Region types metrics are reported for, per base pair (see configs/README.md, Regions):
# all scored bases, bases in any cCRE, in each cCRE class, and outside cCREs
REPORT_STRATA = ["all", "ccre"] + STRATA + ["non_ccre"]
MIN_FIBERS_AUC = 3          # fibers per cell type needed at a bin for the fiber-origin AUC
NUM_PAIRED_PLOTS = 5        # paired dashboards kept (differential tiles with the largest |d|)


def _nan_to_none(x):
    return None if x is None or np.isnan(x) else float(x)


def _meter_summary(meter, output_assays):
    """Per-assay loss (MSE) and Pearson r from a PearsonMeter, plus their means over assays with data."""
    r, mse = meter.value.tolist(), meter.mse.tolist()
    per_assay = {a: {"loss": _nan_to_none(mse[k]), "pearson": _nan_to_none(r[k])} for k, a in enumerate(output_assays)}
    rs = [v["pearson"] for v in per_assay.values() if v["pearson"] is not None]
    ls = [v["loss"] for v in per_assay.values() if v["loss"] is not None]
    return {"n_bp": int(meter.n.max().item()),
            "pearson": float(np.mean(rs)) if rs else None,
            "loss": float(np.mean(ls)) if ls else None,
            "per_assay": per_assay}


def _strata_summary(meters, output_assays):
    return {s: _meter_summary(m, output_assays) for s, m in meters.items() if m.n.max() > 0}


def _partial_summary(pt_meters, bt_meters, pb_meters, output_assays):
    """Per region type and assay: partial correlation of prediction and target given the baseline
    (metrics.partial_correlation, from pooled correlations on the same positions)."""
    out = {}
    for s in pt_meters:
        if pt_meters[s].n.max() == 0:
            continue
        partial = partial_correlation(pt_meters[s].value, bt_meters[s].value, pb_meters[s].value).tolist()
        per_assay = {a: {"pearson": _nan_to_none(partial[k])} for k, a in enumerate(output_assays)}
        rs = [v["pearson"] for v in per_assay.values() if v["pearson"] is not None]
        out[s] = {"n_bp": int(pt_meters[s].n.max().item()), "pearson": float(np.mean(rs)) if rs else None,
                  "loss": None, "per_assay": per_assay}
    return out


def _sel_weights(weights, s):
    """Scored-position weights restricted to a selection: s is None (all), (B, L) per-bp, or (B, K, 1) per assay."""
    if s is None:
        return weights
    s = s.to(weights.dtype)
    return weights * (s[:, None, :] if s.dim() == 2 else s)


def _update_strata(meters, pred, target, weights, selections):
    for name, s in selections.items():
        meters[name].update(pred, target, _sel_weights(weights, s))


def _print_strata_table(cell_type, strata, output_assays, title="Pearson r per base pair, by region type"):
    print(f"[eval] {cell_type}: {title}")
    print(f"  {'region':20s} {'bases':>13s} " + " ".join(f"{a[:8]:>8s}" for a in output_assays) + f" {'mean':>7s}")
    for name, s in strata.items():
        vals = " ".join(f"{'-':>8s}" if s["per_assay"][a]["pearson"] is None else f"{s['per_assay'][a]['pearson']:8.3f}"
                        for a in output_assays)
        mean = f"{'-':>7s}" if s["pearson"] is None else f"{s['pearson']:7.3f}"
        print(f"  {name:20s} {s['n_bp']:>13,} {vals} {mean}")


def _loci_label(name):
    """Report name of a loci set: 'dev/{line}' -> 'loci:dev' (each cell's own set), 'pairs/A__B' -> 'loci:A__B'."""
    return "loci:" + (name.split("/")[0] if "{line}" in name else name.split("/")[-1])


def _auc(a, b):
    """P(a > b) + 0.5 P(a = b) over all pairs (Mann-Whitney), for two 1-d arrays."""
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks over ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


class _CellMeters:
    """Metric accumulators for one cell type."""
    def __init__(self, K, names, output_assays):
        self.composite = AverageMeter()
        self.assay_loss = {a: AverageMeter() for a in output_assays}
        self.strata = {s: PearsonMeter(K) for s in names}
        self.baseline = {s: PearsonMeter(K) for s in names}   # cross-cell-average baseline vs target
        self.pt = {s: PearsonMeter(K) for s in names}         # prediction vs target, on the baseline's positions
        self.pb = {s: PearsonMeter(K) for s in names}         # prediction vs baseline
        self.has_baseline = False
        self.n = 0
        self.records, self.valid = [], 0


class _PairedMeters:
    """Accumulators for the paired readouts of cell types A (first) and B (second)."""
    def __init__(self, K, names):
        self.diff = {s: PearsonMeter(K) for s in names}       # (p_A - p_B) vs (t_A - t_B)
        self.sq = {s: np.zeros((3, K)) for s in names}        # sums of w, w dp^2, w dt^2 (RMS of the differences)
        self.pending = {}                                     # window -> {cell: item} until both cells are in
        self.tiles = []                                       # per-window rows (see _tile_arrays)
        self.auc = []                                         # (tile index, assay, auc, oriented auc, n_a, n_b, d)
        self.plots = []                                       # (|d|, record) for the paired dashboards
        self.n_pairs = 0


class Evaluator:

    def __init__(self, model, dataset, batch_size=1, num_plots_to_log=5, device="cuda", seed=919, num_workers=0,
                 min_coverage=MIN_COVERAGE, output_assays=None, loci=None, paired=False, pair_loci=None):
        self.model = model  # None: score only the baseline (the dataset must carry baseline_bulk)
        self.device = torch.device(device)
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is not None:
            self.model.to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_coverage = min_coverage  # positions covered by fewer fibers are not scored
        self.output_assays = output_assays or self.model.init_args.get("output_assays", ["atac"])

        self.rng = random.Random(seed)
        self.num_to_save = num_plots_to_log

        # Loci sets: name -> {cell type: Loci}; '{line}' in a name means each cell type's own ENCODE line
        self.loci = {}
        cells = getattr(dataset, "cell_type_names", [])
        meta_cells = dataset.metadata.get("cell_types", {})
        for name in (loci or []):
            per_cell = {}
            for ct in cells:
                line = os.path.basename(str((meta_cells.get(ct) or {}).get("encode_dir", ct)).rstrip("/"))
                per_cell[ct] = Loci(dataset.metadata["regions_root"], name.replace("{line}", line))
            self.loci[_loci_label(name)] = (name, per_cell)
        self.report_names = REPORT_STRATA + list(self.loci)

        self.paired = paired
        if paired:
            if model is None or len(cells) != 2 or dataset.__class__.__name__ != "SingleCellFiberDataset":
                raise ValueError("paired eval needs a model and a single-cell eval config with exactly 2 cell types.")
            self.pair_cells = list(cells)
            self.pair_loci = Loci(dataset.metadata["regions_root"], pair_loci) if pair_loci else None
            # Differential readouts: region types and loci sets that are the same for both cells
            self.pair_names = REPORT_STRATA + [n for n, (raw, _) in self.loci.items() if "{line}" not in raw]

    #----------------------------------------------------------------------------------------------
    # Per-sample selections

    def _tile_start(self, chrom, start, end):
        R = self.dataset.regions.region_length
        return (int(start) + int(end)) // 2 - R // 2

    def _selections(self, batch, cells):
        """Per-bp / per-assay position selections per report name: (B, L) bool, (B, K, 1) bool, or None (all)."""
        sel = {"all": None}
        if "strata" in batch:
            st = batch["strata"].to(self.device)        # (B, len(STRATA), L) bool
            in_ccre = st.any(dim=1)
            sel.update({"ccre": in_ccre, "non_ccre": ~in_ccre, **{c: st[:, i] for i, c in enumerate(STRATA)}})
        if self.loci:
            chroms, starts, ends = batch["locus"]
            for label, (_, per_cell) in self.loci.items():
                flags = [per_cell[ct].lookup(c, self._tile_start(c, s, e), self.output_assays)[0] != 0
                         if ct in per_cell else np.zeros(len(self.output_assays), bool)
                         for ct, c, s, e in zip(cells, chroms, starts, ends)]
                sel[label] = torch.from_numpy(np.stack(flags))[:, :, None].to(self.device)
        return sel

    #----------------------------------------------------------------------------------------------
    # Paired readouts

    def _fiber_scores(self, fibers, fmask, rel_bins, bin_size):
        """Per assay with a differential bin: each fiber's mean prediction over the bin (fibers covering >= 1/5 of
        it). fibers (K, L, N) masked per-fiber predictions, fmask (L, N) bool, rel_bins (K,) bin start in the window
        or -1. Returns a list of K arrays (None where the assay has no bin)."""
        out = []
        for k, b in enumerate(rel_bins.tolist()):
            if b < 0 or b + bin_size > fibers.shape[1]:
                out.append(None)
                continue
            cov = fmask[b:b + bin_size].sum(0).float()
            keep = cov >= bin_size / 5
            out.append((fibers[k, b:b + bin_size].float().sum(0)[keep] / cov[keep]).numpy())
        return out

    def _stash_pair(self, pm, i, cell, batch, pred, target, weights, selections, processed_fibers):
        chrom, start, end = batch["locus"][0][i], int(batch["locus"][1][i]), int(batch["locus"][2][i])
        key = (chrom, start)
        d = np.zeros(len(self.output_assays), np.float32)
        rel = np.full(len(self.output_assays), -1, np.int64)
        if self.pair_loci is not None:
            d, bins = self.pair_loci.lookup(chrom, self._tile_start(chrom, start, end), self.output_assays)
            rel = np.where(bins >= 0, bins - start, -1)
        fibers = processed_fibers[i].detach().float().cpu()
        fmask = batch["fiber_mask"][i]
        item = {
            "pred": pred[i:i + 1].float().cpu(), "target": target[i:i + 1].float().cpu(),
            "weights": weights[i:i + 1].float().cpu(),
            "sel": {n: (None if selections[n] is None else selections[n][i:i + 1].cpu()) for n in self.pair_names},
            "scores": self._fiber_scores(fibers, fmask, rel, self.pair_loci.bin_size) if self.pair_loci else None,
            "d": d, "rel_bins": rel, "locus": (chrom, start, end),
        }
        # Keep fibers only for windows that could make it into the paired dashboards
        max_d = float(np.abs(d).max())
        if max_d > 0 and (len(pm.plots) < NUM_PAIRED_PLOTS or max_d > pm.plots[-1][0]):
            n = int(batch["n_fibers"][i])
            item["fibers"], item["fiber_mask"] = fibers[:, :, :n], fmask[:, :n]
        slot = pm.pending.setdefault(key, {})
        slot[cell] = item
        if len(slot) == 2:
            self._complete_pair(pm, pm.pending.pop(key))

    def _complete_pair(self, pm, slot):
        A, B = (slot[c] for c in self.pair_cells)
        w = A["weights"] * B["weights"]                        # scored in both cells
        dp, dt = A["pred"] - B["pred"], A["target"] - B["target"]
        for name in self.pair_names:
            ws = _sel_weights(w, A["sel"][name])
            pm.diff[name].update(dp, dt, ws)
            pm.sq[name] += np.stack([ws.sum(dim=(0, 2)).double().numpy(),
                                     (ws * dp ** 2).sum(dim=(0, 2)).double().numpy(),
                                     (ws * dt ** 2).sum(dim=(0, 2)).double().numpy()])
        pm.n_pairs += 1

        # Per-window row: mean differences over the window and over the differential bin (scored in both)
        K, bin_size = dp.shape[1], (self.pair_loci.bin_size if self.pair_loci else 0)
        n_w = w[0].sum(-1)
        row = {"chrom": A["locus"][0], "start": A["locus"][1], "d": A["d"],
               "dp_win": ((w[0] * dp[0]).sum(-1) / n_w.clamp(min=1)).numpy(),
               "dt_win": ((w[0] * dt[0]).sum(-1) / n_w.clamp(min=1)).numpy(), "n_win": n_w.numpy(),
               "dp_bin": np.zeros(K, np.float32), "dt_bin": np.zeros(K, np.float32), "n_bin": np.zeros(K, np.float32)}
        for k, b in enumerate(A["rel_bins"].tolist()):
            if b >= 0:
                wb = w[0, k, b:b + bin_size]
                n = float(wb.sum())
                row["n_bin"][k] = n
                if n > 0:
                    row["dp_bin"][k] = float((wb * dp[0, k, b:b + bin_size]).sum()) / n
                    row["dt_bin"][k] = float((wb * dt[0, k, b:b + bin_size]).sum()) / n
        tile_idx = len(pm.tiles)
        pm.tiles.append(row)

        # Fiber-origin AUC at the differential bins
        if A["scores"] is not None:
            for k in range(K):
                sa, sb = A["scores"][k], B["scores"][k]
                if sa is None or len(sa) < MIN_FIBERS_AUC or len(sb) < MIN_FIBERS_AUC:
                    continue
                auc = _auc(sa, sb)
                d = float(A["d"][k])
                pm.auc.append((tile_idx, k, auc, auc if d > 0 else 1.0 - auc, len(sa), len(sb), d))

        # Paired dashboards: keep the windows with the largest |d|
        if "fibers" in A and "fibers" in B:
            max_d = float(np.abs(A["d"]).max())
            if len(pm.plots) < NUM_PAIRED_PLOTS or max_d > pm.plots[-1][0]:
                rec = {"locus": A["locus"], "cells": self.pair_cells, "d": A["d"], "rel_bins": A["rel_bins"],
                       "bin_size": bin_size,
                       "pred": [A["pred"][0], B["pred"][0]], "target": [A["target"][0], B["target"][0]],
                       "score": [A["weights"][0] > 0, B["weights"][0] > 0],
                       "fibers": [A["fibers"], B["fibers"]], "fiber_mask": [A["fiber_mask"], B["fiber_mask"]]}
                pm.plots = sorted(pm.plots + [(max_d, rec)], key=lambda x: -x[0])[:NUM_PAIRED_PLOTS]

    def _paired_summary(self, pm):
        assays = self.output_assays
        K = len(assays)
        strata = {}
        for name in self.pair_names:
            m = pm.diff[name]
            if m.n.max() == 0:
                continue
            r = m.value.tolist()
            n, sp, st = pm.sq[name]
            rms_p, rms_t = np.sqrt(sp / np.maximum(n, 1)), np.sqrt(st / np.maximum(n, 1))
            per_assay = {a: {"diff_pearson": _nan_to_none(r[k]),
                             "rms_dp": float(rms_p[k]) if n[k] > 0 else None,
                             "rms_dt": float(rms_t[k]) if n[k] > 0 else None} for k, a in enumerate(assays)}
            rs = [v["diff_pearson"] for v in per_assay.values() if v["diff_pearson"] is not None]
            strata[name] = {"n_bp": int(m.n.max().item()), "diff_pearson": float(np.mean(rs)) if rs else None,
                            "per_assay": per_assay}

        tiles = self._tile_arrays(pm)
        loci = {}
        if self.pair_loci is not None and len(pm.tiles):
            auc = np.array(pm.auc, dtype=np.float64).reshape(-1, 7)
            for k, a in enumerate(assays):
                has = (tiles["d"][:, k] != 0) & (tiles["n_bin"][:, k] > 0)
                sign_acc = float(np.mean(np.sign(tiles["dp_bin"][has, k]) == np.sign(tiles["d"][has, k]))) if has.any() else None
                ak = auc[auc[:, 1] == k] if len(auc) else auc
                loci[a] = {"n_tiles": int(has.sum()), "sign_accuracy": sign_acc,
                           "n_auc_tiles": int(len(ak)),
                           "fiber_auc": float(ak[:, 3].mean()) if len(ak) else None,
                           "fiber_auc_abs_dev": float(np.abs(ak[:, 2] - 0.5).mean()) if len(ak) else None}
        return {"cells": self.pair_cells, "n_pairs": pm.n_pairs, "strata": strata,
                "pair_loci": self.pair_loci.name if self.pair_loci else None, "loci": loci,
                "tiles": tiles, "auc": np.array(pm.auc, dtype=np.float64).reshape(-1, 7),
                "plots": [rec for _, rec in pm.plots]}

    @staticmethod
    def _tile_arrays(pm):
        if not pm.tiles:
            return {}
        out = {k: np.stack([t[k] for t in pm.tiles]) for k in ("d", "dp_win", "dt_win", "n_win", "dp_bin", "dt_bin", "n_bin")}
        out["chrom"] = np.array([t["chrom"] for t in pm.tiles])
        out["start"] = np.array([t["start"] for t in pm.tiles], dtype=np.int64)
        return out

    #----------------------------------------------------------------------------------------------

    def evaluate(self, save_path=None):
        has_model = self.model is not None
        if has_model:
            self.model.eval()

        test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            worker_init_fn=seed_worker,
            drop_last=False,
            num_workers=self.num_workers,  # the dataset strides its regions across workers
            multiprocessing_context=loader_context(self.num_workers),
        )

        output_assays = self.output_assays
        K = len(output_assays)
        meters = {}                    # cell type -> _CellMeters
        pm = _PairedMeters(K, self.pair_names) if self.paired else None
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                fiber_features, target_bulk, forward_kwargs = unpack_batch(batch, self.device)
                assay_mask = batch["assay_mask"].to(self.device)
                coverage = batch["fiber_coverage"].to(self.device)
                B = fiber_features.shape[0]
                num_samples += B
                ct = batch.get("cell_type", ["Unknown"] * B)
                cells = list(ct) if isinstance(ct, (list, tuple)) else [ct] * B

                # Scored entries only: assays the cell type has, at positions covered by >= min_coverage fibers
                weights = score_weights(assay_mask, coverage, self.min_coverage)    # (B, K, L)
                selections = self._selections(batch, cells)

                pred_composite_bulk, processed_fibers = None, None
                if has_model:
                    with torch.amp.autocast(self.device_type, dtype=torch.float16):
                        pred_composite_bulk, processed_fibers = self.model(fiber_features, **forward_kwargs)
                    pred_composite_bulk = pred_composite_bulk.float()

                for i, cell in enumerate(cells):
                    cm = meters.setdefault(cell, _CellMeters(K, self.report_names, output_assays))
                    cm.n += 1
                    sl = slice(i, i + 1)
                    sel_i = {n: (None if s is None else s[sl]) for n, s in selections.items()}
                    w_i, t_i = weights[sl], target_bulk[sl]
                    if has_model:
                        p_i = pred_composite_bulk[sl]
                        comp_loss = masked_mse(p_i, t_i, w_i).item()
                        cm.composite.update(comp_loss)
                        losses, counts = per_assay_mse(p_i, t_i, w_i)
                        for k, assay in enumerate(output_assays):
                            if counts[k] > 0:
                                cm.assay_loss[assay].update(losses[k].item(), n=int(counts[k].item()))
                        _update_strata(cm.strata, p_i, t_i, w_i, sel_i)

                    if "baseline_bulk" in batch:
                        cm.has_baseline = True
                        baseline = batch["baseline_bulk"][sl].to(self.device)
                        # Also requires a training cell line with the assay (baseline_mask)
                        b_w = w_i * batch["baseline_mask"][sl].to(self.device)[:, :, None]
                        _update_strata(cm.baseline, baseline, t_i, b_w, sel_i)
                        if has_model:
                            _update_strata(cm.pt, p_i, t_i, b_w, sel_i)
                            _update_strata(cm.pb, p_i, baseline, b_w, sel_i)

                    if not has_model:
                        continue

                    if self.paired:
                        self._stash_pair(pm, i, cell, batch, pred_composite_bulk, target_bulk, weights, selections,
                                         processed_fibers)

                    record = {
                        "locus": [[batch["locus"][0][i]], batch["locus"][1][sl], batch["locus"][2][sl]],
                        "cell_type": cell,
                        "fiber_features": fiber_features[sl].cpu(),
                        "processed_fibers": processed_fibers[sl].cpu(),
                        "pred_bulk": p_i.cpu(),
                        "target_bulk": t_i.cpu(),
                        "assay_mask": assay_mask[sl].cpu(),
                        "score_mask": (coverage[sl] >= self.min_coverage).cpu(),
                        "loss": comp_loss,
                    }
                    cm.valid += 1
                    if len(cm.records) < self.num_to_save:
                        cm.records.append(record)
                    else:
                        j = self.rng.randint(0, cm.valid - 1)
                        if j < self.num_to_save:
                            cm.records[j] = record

                    if save_path is not None:
                        fig = plot_evaluator_record_t(
                            record_t=record,
                            input_flags=self.model.init_args["input_flags"],
                            assay_avg_losses=None,
                            output_assays=output_assays,
                            mode="Test"
                        )
                        plt.savefig(f"{save_path}test_e_{batch_idx}_{cell}.png")
                        plt.close()

        expected = getattr(self.dataset, "expected_samples", None)
        n_cells = max(len(getattr(self.dataset, "cell_type_names", [])), 1)
        results = {"num_locus": num_samples, "expected_samples": expected, "cells": {}}
        for cell, cm in meters.items():
            res = {"num_locus": cm.n, "locus_records": cm.records,
                   "expected_samples": expected if cell == "mixed" or expected is None else expected // n_cells}
            if has_model:
                # None = the evaluated cell type has no bulk data for that assay (or r undefined: flat signal)
                strata = _strata_summary(cm.strata, output_assays)
                all_bp = strata.get("all", {"pearson": None, "per_assay": {a: {"pearson": None} for a in output_assays}})
                res.update({
                    # composite pearson = mean of the per-assay r over assays with data
                    "composite": {"loss": cm.composite.avg, "pearson": all_bp["pearson"]},
                    "per_assay": {a: {"loss": cm.assay_loss[a].avg if cm.assay_loss[a].count > 0 else None,
                                      "pearson": all_bp["per_assay"][a]["pearson"]} for a in output_assays},
                    "strata": strata,
                })
            if cm.has_baseline:
                b_strata = _strata_summary(cm.baseline, output_assays)
                b_all = b_strata.get("all", {"pearson": None, "loss": None, "per_assay": {}})
                res["baseline"] = {"composite": {"loss": b_all["loss"], "pearson": b_all["pearson"]},
                                   "per_assay": b_all["per_assay"], "strata": b_strata}
                if has_model:
                    res["partial"] = {"strata": _partial_summary(cm.pt, cm.baseline, cm.pb, output_assays)}
            results["cells"][cell] = res
        if self.paired:
            results["paired"] = self._paired_summary(pm)
        return results


def resolve_baseline_cells(train_cells, metapaths, output_assays):
    """Metadata entries ({cell: {"bulk", "encode_dir", ...}}) for the cells a model was trained on."""
    return resolve_config_with_metapaths({"cell_types": {c: None for c in train_cells}}, metapaths,
                                         output_assays)["cell_types"]


def _print_paired(name, paired, output_assays):
    a, b = paired["cells"]
    print(f"[eval] {name}: paired readouts, {a} (A) vs {b} (B), {paired['n_pairs']} windows with both")
    print(f"  differential r = corr(p_A - p_B, t_A - t_B); RMS of p_A - p_B / t_A - t_B on the same positions")
    print(f"  {'region':20s} {'bases':>13s} " + " ".join(f"{x[:8]:>8s}" for x in output_assays) + f" {'mean':>7s}")
    for s_name, s in paired["strata"].items():
        vals = " ".join(f"{'-':>8s}" if s["per_assay"][x]["diff_pearson"] is None else f"{s['per_assay'][x]['diff_pearson']:8.3f}"
                        for x in output_assays)
        mean = f"{'-':>7s}" if s["diff_pearson"] is None else f"{s['diff_pearson']:7.3f}"
        print(f"  {s_name:20s} {s['n_bp']:>13,} {vals} {mean}")
    s = paired["strata"].get("all")
    if s:
        print(f"  {'RMS dp / dt (all)':34s} " + " ".join(
            f"{'-':>8s}" if s["per_assay"][x]["rms_dp"] is None else f"{s['per_assay'][x]['rms_dp']:.2f}/{s['per_assay'][x]['rms_dt']:.2f}".rjust(8)
            for x in output_assays))
    if paired["loci"]:
        print(f"  At the differential bins of {paired['pair_loci']} (sign accuracy; fiber-origin AUC oriented by d; "
              f"mean |AUC - 0.5|):")
        for row, key in (("tiles", "n_tiles"), ("sign accuracy", "sign_accuracy"), ("AUC tiles", "n_auc_tiles"),
                         ("fiber AUC", "fiber_auc"), ("|AUC - 0.5|", "fiber_auc_abs_dev")):
            vals = []
            for x in output_assays:
                v = paired["loci"].get(x, {}).get(key)
                vals.append(f"{'-':>8s}" if v is None else (f"{v:8d}" if isinstance(v, int) else f"{v:8.3f}"))
            print(f"  {row:34s} " + " ".join(vals))


def run_final_eval(model, eval_configs, train_args, wandb_run, device, plots_output_dir=None,
                   baseline_cells=None, output_assays=None, output_dir=None, baseline_include_own=False):
    """
    Run evaluation over a list of eval config paths.
    Logs to wandb_run immediately after each config (if wandb_run is not None).
    baseline_cells: names of the cells the model was trained on; adds the cross-cell-average baseline and the
    cell-specific (partial) correlation. model=None scores only the baseline (then output_assays is required).
    baseline_include_own: the baseline includes the evaluated cell's own line (mixture reference, T3).
    output_dir: where paired readouts save their per-window arrays (.npz) and figures.
    Returns a summary dict keyed by cell type (by '<config>/<cell type>' for configs with several cell types,
    plus '<config>/paired' for paired readouts).
    """
    has_model = model is not None
    ref_name = "mixture reference" if baseline_include_own else "cross-cell average"
    print("\n" + "=" * 60)
    print(f" Running Final Evaluation ({len(eval_configs)} config(s))"
          + (f" with the baseline ({ref_name})" if baseline_cells and has_model else
             f" of the baseline ({ref_name}) only" if baseline_cells else "") + "...")
    print("=" * 60)

    metapaths = None
    if hasattr(train_args, "metapaths") and os.path.exists(train_args.metapaths):
        metapaths = load_metapaths(train_args.metapaths)

    if has_model:
        output_assays = model.init_args.get("output_assays", ["atac"])
        input_flags, dna_type = model.init_args["input_flags"], model.init_args["dna_type"]
    else:
        if not (baseline_cells and output_assays):
            raise ValueError("Without a model, run_final_eval needs baseline_cells and output_assays.")
        input_flags, dna_type = [1, 1, 1, 1, 1], "none"   # fibers are still read: they decide the scored positions

    baseline_meta = None
    if baseline_cells:
        if metapaths is None:
            raise ValueError("The baseline needs metapaths to find the training cells' tracks.")
        baseline_meta = resolve_baseline_cells(baseline_cells, metapaths, output_assays)
        print(f"[eval] Baseline: mean of the training cells {list(baseline_cells)} "
              + ("(including the evaluated cell line: mixture reference)" if baseline_include_own
                 else "(excluding the evaluated cell line)"))
    summary = {}

    for eval_config_path in eval_configs:
        eval_cfg = load_config_file(eval_config_path)
        config_name = os.path.splitext(os.path.basename(eval_config_path))[0]
        dataset_type = eval_cfg.get("dataset_type", "single")

        cell_types_dict = eval_cfg.get("cell_types", {})
        print(f"\n[eval] Cell type(s): {', '.join(cell_types_dict) or 'unknown'} | Config: {eval_config_path}")

        if metapaths is not None:
            eval_metadata = resolve_config_with_metapaths(eval_cfg, metapaths, output_assays)
        else:
            eval_metadata = eval_cfg.get("metadata", {})

        windows = eval_cfg.get("windows")
        if not windows:
            raise ValueError(f"{eval_config_path}: eval configs need 'windows: <window set name>' "
                             "(e.g. test_chr21_mini2000); see configs/README.md, Regions.")
        eval_seed = eval_cfg.get("seed", 919)
        ds_kwargs = dict(
            metadata=eval_metadata,
            fibers_per_entry=eval_cfg.get("fibers_per_entry", getattr(train_args, "fibers_per_entry", 200)),
            context_length=eval_cfg.get("context_length", getattr(train_args, "context_length", 4096)),
            windows=windows,
            max_regions=eval_cfg.get("max_regions"),
            input_flags=input_flags,
            dna_type=dna_type,
            output_assays=output_assays,
            seed=eval_seed,
        )
        if dataset_type == "mixed":
            ds_kwargs["mix_range"] = eval_cfg.get("mix_range")
        else:
            ds_kwargs.update(baseline_cells=baseline_meta, baseline_include_own=baseline_include_own)
        test_set = make_fiber_dataset(dataset_type, mode="eval", **ds_kwargs)

        save_path = None
        if plots_output_dir is not None and has_model:
            save_path = os.path.join(plots_output_dir, config_name) + "/"
            os.makedirs(save_path, exist_ok=True)

        paired = bool(eval_cfg.get("paired", False))
        evaluator = Evaluator(model, test_set, batch_size=1, num_plots_to_log=5, device=device, seed=eval_seed,
                              num_workers=getattr(train_args, "num_workers", 0),
                              min_coverage=getattr(train_args, "min_coverage", MIN_COVERAGE),
                              output_assays=output_assays, loci=eval_cfg.get("loci"),
                              paired=paired, pair_loci=eval_cfg.get("pair_loci"))
        eval_results = evaluator.evaluate(save_path=save_path)
        multi = len(eval_results["cells"]) > 1 or paired

        for cell_type, res in eval_results["cells"].items():
            key = f"{config_name}/{cell_type}" if multi else cell_type
            summary[key] = _report_cell(key, res, eval_config_path, windows, output_assays, has_model,
                                        baseline_cells, baseline_include_own, wandb_run, test_set.input_flags)

        if paired:
            p = eval_results["paired"]
            _print_paired(config_name, p, output_assays)
            entry = {k: v for k, v in p.items() if k not in ("tiles", "auc", "plots")}
            summary[f"{config_name}/paired"] = entry
            figs = {}
            if p["tiles"]:
                figs["differential_scatter"] = plot_differential_scatter(p["tiles"], output_assays, p["cells"])
            if len(p["auc"]):
                figs["fiber_auc"] = plot_fiber_auc(p["auc"], output_assays, p["cells"])
            for j, rec in enumerate(p["plots"]):
                figs[f"paired_locus_{j}"] = plot_paired_locus(rec, output_assays)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                np.savez_compressed(os.path.join(output_dir, f"{config_name}_paired.npz"),
                                    auc=p["auc"], auc_columns=np.array(["tile", "assay", "auc", "auc_oriented",
                                                                        "n_a", "n_b", "d"]),
                                    assays=np.array(output_assays), cells=np.array(p["cells"]),
                                    **{f"tile_{k}": v for k, v in p["tiles"].items()})
                for fname, fig in figs.items():
                    fig.savefig(os.path.join(output_dir, f"{config_name}_{fname}.png"), dpi=110, bbox_inches="tight")
                print(f"[eval] Paired arrays and figures saved to {output_dir}/{config_name}_*")
            if wandb_run is not None:
                prefix = f"test/{config_name}/paired"
                log = {}
                for s_name, s in p["strata"].items():
                    if s["diff_pearson"] is not None:
                        log[f"{prefix}/diff_pearson" + ("" if s_name == "all" else f"@{s_name}")] = s["diff_pearson"]
                for a, v in p["loci"].items():
                    for kname in ("sign_accuracy", "fiber_auc"):
                        if v[kname] is not None:
                            log[f"{prefix}/{kname}_{a}"] = v[kname]
                for fname, fig in figs.items():
                    log[f"Evaluation/{config_name}_{fname}"] = wandb.Image(fig)
                wandb_run.log(log)
            for fig in figs.values():
                plt.close(fig)

    print(f"\n Evaluation complete for {len(eval_configs)} config(s).")
    return summary


def _report_cell(name, res, eval_config_path, windows, output_assays, has_model, baseline_cells,
                 baseline_include_own, wandb_run, input_flags):
    """Print, log and return the summary entry for one cell type of an eval config."""
    n_done, n_expected = res["num_locus"], res["expected_samples"]
    entry = {"eval_config": eval_config_path, "windows": windows, "num_loci": n_done, "expected_samples": n_expected}
    skipped = f" ({n_expected - n_done} skipped: no fibers)" if n_expected is not None else ""
    line = f"[eval] {name} on {windows}: {n_done} of {n_expected} samples evaluated{skipped}"
    ref_name = "mixture reference" if baseline_include_own else "cross-cell average"

    if has_model:
        composite_loss = res["composite"]["loss"]
        composite_r = res["composite"]["pearson"]
        r_str = f"{composite_r:.4f}" if composite_r is not None else "n/a"
        print(f"{line} | composite loss: {composite_loss:.6f} | pearson: {r_str}")
        _print_strata_table(name, res["strata"], output_assays)
        assay_avg = {a: res["per_assay"][a]["loss"] for a in output_assays}
        assay_avg_r = {a: res["per_assay"][a]["pearson"] for a in output_assays}
        entry.update({
            "composite_loss": composite_loss,
            "pearson": composite_r,
            "per_assay_loss": assay_avg,
            "per_assay_pearson": assay_avg_r,
            "strata": res["strata"],   # bp-level loss / r per region type, per assay
        })
    else:
        print(line)

    if "baseline" in res:
        b = res["baseline"]
        b_r = f"{b['composite']['pearson']:.4f}" if b["composite"]["pearson"] is not None else "n/a"
        b_l = f"{b['composite']['loss']:.6f}" if b["composite"]["loss"] is not None else "n/a"
        print(f"[eval] {name} baseline ({ref_name}) | loss: {b_l} | pearson: {b_r}")
        _print_strata_table(name, b["strata"], output_assays, f"BASELINE ({ref_name}) Pearson r by region type")
        entry["baseline"] = {"train_cells": list(baseline_cells), "include_own": baseline_include_own, **b}
        if "partial" in res:
            _print_strata_table(name, res["partial"]["strata"], output_assays,
                                f"cell-specific r: partial correlation of prediction and target given the {ref_name}")
            entry["partial"] = res["partial"]

    if wandb_run is not None:
        prefix = f"test/{name}"
        ct_log_dict = {}
        if has_model:
            ct_log_dict[f"{prefix}/loss"] = composite_loss
            if composite_r is not None:
                ct_log_dict[f"{prefix}/pearson"] = composite_r
            for assay, data in res.get("per_assay", {}).items():
                if data["loss"] is not None:
                    ct_log_dict[f"{prefix}/loss_{assay}"] = data["loss"]
                if data["pearson"] is not None:
                    ct_log_dict[f"{prefix}/pearson_{assay}"] = data["pearson"]
            # Per region type: means over assays (per-assay detail is in the summary JSON)
            for s_name, s in res["strata"].items():
                if s_name == "all":
                    continue
                if s["pearson"] is not None:
                    ct_log_dict[f"{prefix}/pearson@{s_name}"] = s["pearson"]
                if s["loss"] is not None:
                    ct_log_dict[f"{prefix}/loss@{s_name}"] = s["loss"]
        for kind in ("baseline", "partial"):
            for s_name, s in res.get(kind, {}).get("strata", {}).items():
                if s["pearson"] is not None:
                    ct_log_dict[f"{prefix}/{kind}_pearson" + ("" if s_name == "all" else f"@{s_name}")] = s["pearson"]

        if has_model:
            wandb_image_list = []
            for idx, record in enumerate(res.get("locus_records", [])):
                fig = plot_evaluator_record_t(
                    record_t=record,
                    input_flags=input_flags,
                    assay_avg_losses=assay_avg,
                    output_assays=output_assays,
                    mode="Test",
                    assay_avg_pearson=assay_avg_r,
                )
                chr_name = record["locus"][0][0]
                start = record["locus"][1][0]
                end = record["locus"][2][0]
                caption = f"[{name}] Locus {idx}/{res['num_locus']}: {chr_name}:{start}-{end}"
                wandb_image_list.append(wandb.Image(fig, caption=caption))
                plt.close(fig)
            ct_log_dict[f"Evaluation/{name}_Dashboards"] = wandb_image_list
        wandb_run.log(ct_log_dict)
        if has_model:
            print(f"[eval] Logged {len(wandb_image_list)} dashboards for {name} to WandB.")
    return entry
