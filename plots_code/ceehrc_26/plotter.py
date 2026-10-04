"""
One-off script for the ceehrc_26 poster/talk.

Rather than plotting all ~11,373 HepG2_200U chr21 cCREs (too slow to run, too
many images to sift through), this first does a cheap pre-filter over the
ground-truth bulk signal alone (bigWig lookup only, no fiber-read gathering)
and keeps only loci where ATAC, H3K4me3, AND H3K27ac all peak above 1.7
(asinh-space, same units shown in the dashboards; H3K27me3 ignored — rarely
present in cCREs). Only the much smaller filtered set then gets the expensive
fiber-gathering + model inference, for three models:
  - hep_only:  trained only on HepG2_200U
  - gm_k5:     trained on GM12878 + K562_200U (never saw HepG2_200U)
  - all_three: trained on GM12878 + K562_200U + HepG2_200U

The filter pass is independent of any checkpoint (it only reads HepG2_200U's
own ground-truth bigWigs), so it runs once and the same locus set is reused
for all three models — meaning the three model directories end up with
directly comparable plots of the exact same loci.

All three checkpoints share the same preprocessing config (input_flags,
dna_type, output_assays — verified at runtime), so per locus the fiber
tensor (the actual bottleneck: CRAM decoding + per-fiber feature extraction)
is built ONCE and reused across all three models' forward passes, instead of
rebuilding it redundantly per model.

Output goes to ignore/ceehrc_26/<model_label>/, one PNG dashboard per locus,
named "<index>_<chrom>_<start>-<end>.png".

Usage (from anywhere, paths are resolved relative to the repo root):
    python plots_code/ceehrc_26/plotter.py
"""

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from models import BaseModel
from data_utils import make_fiber_dataset
from utils import load_config_file, load_metapaths, resolve_config_with_metapaths, unpack_batch, set_seed
from vis_utils import plot_evaluator_record_t

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "evals", "eval15_HepG2_200U.yaml")
METAPATHS_PATH = os.path.join(REPO_ROOT, "configs", "metapaths.yaml")
OUT_ROOT = os.path.join(REPO_ROOT, "ignore", "ceehrc_26")

MODELS = {
    "hep_only": os.path.join(
        REPO_ROOT, "results",
        "26-09-20_T21-33-58_All_A_Hep_run_data15_HepG2_200U_model11_uct_mid_all_A_train01",
        "Model_epoch_25.pt",
    ),
    "gm_k5": os.path.join(
        REPO_ROOT, "results",
        "26-09-21_T05-57-36_All_A_GM_K5_run_data16_GM12878_K562_200U_model11_uct_mid_all_A_train01",
        "Model_epoch_25.pt",
    ),
    "all_three": os.path.join(
        REPO_ROOT, "results",
        "26-09-22_T11-37-41_All_A_GM_K5_Hep_run_data17_GM12878_K562_200U_HepG2_200U_model11_uct_mid_all_A_train01",
        "Model_epoch_25.pt",
    ),
}

# Filter-pass config. These match model11_uct_mid_all_A.yaml (used by all 3
# checkpoints above) — only used to construct the probe dataset that reads
# ground-truth bigWigs; irrelevant to the (skipped) fiber-gathering step.
SIGNAL_THRESHOLD = 1.7
FILTER_ASSAYS = ["atac", "h3k4me3", "h3k27ac"]
OUTPUT_ASSAYS = ["atac", "h3k4me3", "h3k27ac", "h3k27me3"]
PROBE_INPUT_FLAGS = [1, 1, 1, 1, 1]
PROBE_DNA_TYPE = "ref"


def build_dataset(output_assays, input_flags, dna_type):
    eval_cfg = load_config_file(EVAL_CONFIG_PATH)
    metapaths = load_metapaths(METAPATHS_PATH)
    metadata = resolve_config_with_metapaths(eval_cfg, metapaths, output_assays)

    seed = eval_cfg.get("seed", 919)
    num_sample_ccres = eval_cfg.get("num_sample_ccres", -1)
    set_seed(seed)

    return make_fiber_dataset(
        eval_cfg.get("dataset_type", "single"),
        mode="eval",
        metadata=metadata,
        fibers_per_entry=eval_cfg.get("fibers_per_entry", 200),
        context_length=eval_cfg.get("context_length", 4096),
        iters_per_epoch=num_sample_ccres,
        num_sample_ccres=num_sample_ccres,
        input_flags=input_flags,
        dna_type=dna_type,
        output_assays=output_assays,
        seed=seed,
    )


def build_filtered_ccres():
    """
    Cheap pre-filter over ground-truth bulk signal only (no fiber gathering).
    Keeps a raw (chrom, start, end) cCRE entry iff ATAC, H3K4me3, and H3K27ac
    all peak above SIGNAL_THRESHOLD somewhere in the expanded context window.
    Independent of model checkpoint — HepG2_200U ground truth is fixed.
    """
    dataset = build_dataset(OUTPUT_ASSAYS, PROBE_INPUT_FLAGS, PROBE_DNA_TYPE)
    dataset.init_worker_resources()

    assay_idx = {a: OUTPUT_ASSAYS.index(a) for a in FILTER_ASSAYS}
    total = len(dataset.ccre_list)
    kept = []

    start_time = time.time()
    for i, entry in enumerate(dataset.ccre_list):
        locus = dataset.expand_ccre_locus(*entry, jitter_range=0)
        sig = dataset.get_other_bw_data(0, *locus)
        if torch.isnan(sig).any():
            continue
        if all(sig[assay_idx[a]].max().item() > SIGNAL_THRESHOLD for a in FILTER_ASSAYS):
            kept.append(tuple(entry))

        if (i + 1) % 2000 == 0:
            print(f"  filter: scanned {i + 1}/{total}, {len(kept)} passing so far "
                  f"({time.time() - start_time:.0f}s elapsed)", flush=True)

    print(f"Filter pass: {len(kept)} / {total} cCREs passed "
          f"({' & '.join(FILTER_ASSAYS)} > {SIGNAL_THRESHOLD}), "
          f"{time.time() - start_time:.0f}s total", flush=True)
    return kept


def load_models():
    """Load all 3 checkpoints and verify they share the preprocessing config
    the shared-fiber-tensor optimization below depends on."""
    models = {}
    for label, ckpt in MODELS.items():
        print(f"Loading [{label}] {ckpt}", flush=True)
        model, _ = BaseModel.load_model(ckpt, map_location=DEVICE)
        model.to(DEVICE)
        model.eval()
        models[label] = model

    ref_label = next(iter(models))
    ref = models[ref_label]
    for label, model in models.items():
        assert model.init_args["input_flags"] == ref.init_args["input_flags"], \
            f"input_flags mismatch: [{label}] vs [{ref_label}] — can't share fiber tensors"
        assert model.init_args["dna_type"] == ref.init_args["dna_type"], \
            f"dna_type mismatch: [{label}] vs [{ref_label}] — can't share fiber tensors"
        assert model.init_args.get("output_assays") == ref.init_args.get("output_assays"), \
            f"output_assays mismatch: [{label}] vs [{ref_label}] — can't share target_bulk"

    return models


def plot_all_models(models, filtered_ccres):
    """
    Single pass over the filtered loci. Per locus, the fiber tensor is built
    once (the bottleneck: CRAM decoding + per-fiber feature extraction) and
    reused across all 3 models' forward passes, instead of rebuilding it
    redundantly once per model.
    """
    ref_model = next(iter(models.values()))
    output_assays = ref_model.init_args.get("output_assays", ["atac"])
    input_flags = ref_model.init_args["input_flags"]
    dna_type = ref_model.init_args["dna_type"]

    dataset = build_dataset(output_assays, input_flags, dna_type)
    dataset.ccre_list = filtered_ccres
    dataset.num_sample_ccres = -1

    out_dirs = {}
    for label in models:
        out_dirs[label] = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dirs[label], exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, drop_last=False)
    criterion = torch.nn.MSELoss()

    total_loci = len(filtered_ccres)
    plotted = {label: 0 for label in models}
    skipped = {label: 0 for label in models}
    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            chrom0 = batch["locus"][0][0]
            start0 = int(batch["locus"][1][0])
            end0 = int(batch["locus"][2][0])
            fname = f"{i:05d}_{chrom0}_{start0}-{end0}.png"
            out_paths = {label: os.path.join(out_dirs[label], fname) for label in models}

            # Skip building the fiber tensor entirely if every model already has this locus.
            if all(os.path.exists(p) for p in out_paths.values()):
                for label in models:
                    skipped[label] += 1
                continue

            fiber_features, target_bulk, forward_kwargs = unpack_batch(batch, DEVICE)

            for label, model in models.items():
                out_path = out_paths[label]
                if os.path.exists(out_path):
                    skipped[label] += 1
                    continue

                with torch.amp.autocast(DEVICE_TYPE, dtype=torch.float16):
                    pred_bulk, processed_fibers = model(fiber_features, **forward_kwargs)
                loss = criterion(pred_bulk, target_bulk).item()

                record_t = {
                    "locus": batch["locus"],
                    "cell_type": batch.get("cell_type", ["HepG2_200U"]),
                    "fiber_features": fiber_features.cpu(),
                    "processed_fibers": processed_fibers.cpu(),
                    "pred_bulk": pred_bulk.cpu(),
                    "target_bulk": target_bulk.cpu(),
                    "loss": loss,
                }

                fig = plot_evaluator_record_t(
                    record_t=record_t,
                    input_flags=input_flags,
                    assay_avg_losses=None,
                    output_assays=output_assays,
                    mode="Test",
                )
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                plotted[label] += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  locus {i + 1}/{total_loci} — {elapsed / (i + 1):.2f}s/locus avg, "
                      f"{elapsed / 60:.1f} min elapsed", flush=True)

    total_time = time.time() - start_time
    for label in models:
        print(f"=== [{label}] done: {plotted[label]} plotted, {skipped[label]} already existed, "
              f"saved to {out_dirs[label]} ===", flush=True)
    print(f"Total: {total_time / 60:.1f} min for {total_loci} loci x {len(models)} models", flush=True)


if __name__ == "__main__":
    print(f"Repo root: {REPO_ROOT}")
    print(f"Output root: {OUT_ROOT}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    filtered_ccres = build_filtered_ccres()
    models = load_models()
    plot_all_models(models, filtered_ccres)

    print("\nAll models done.", flush=True)
