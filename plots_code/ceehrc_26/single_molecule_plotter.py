"""
One-off script for the ceehrc_26 poster/talk.

For a hand-picked list of indices (the <index> prefix from filenames already
reviewed under ignore/ceehrc_26/<model_label>/), for each of the 3 models:
  1. Copy the existing composite dashboard PNG from ignore/ceehrc_26/<label>/
     to plots/ceehrc_26/<label>/ (plots/ is mutagen-synced, unlike ignore/).
  2. Re-run inference for that locus (processed_fibers was never saved, only
     the rendered PNG) and render one combined single-molecule figure via the
     existing vis_utils.plot_single_column_fiber_stack_t, generalized to take
     a list per assay: one column per assay (4 columns), each column being
     [bulk comparison] then per selected fiber [fiber-seq input row, imputed
     signal row]. Fibers are filtered to those with summed m6a >
     M6A_THRESHOLD, and the same fiber set is shared across all 4 columns.
     Saved as "<index>_..._single_molecule.png".

Edit SELECTED_INDICES below (no CLI by design), then review this script and
plot_hepg2_chr21.sh before submitting — the slurm script is NOT submitted by
this change.

Usage: python plots_code/ceehrc_26/single_molecule_plotter.py
"""

import os
import sys
import glob
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import plotter` works
from plotter import REPO_ROOT, OUT_ROOT, DEVICE, DEVICE_TYPE, build_dataset, build_filtered_ccres, load_models

from utils import unpack_batch
from vis_utils import plot_single_column_fiber_stack_t

# ---- EDIT ME ----
# Indices to promote — the <index> prefix of files under ignore/ceehrc_26/<label>/.
SELECTED_INDICES = [147,165,473,720,868,]
NUM_FIBERS = 10
M6A_THRESHOLD = 200
# -----------------

PLOTS_ROOT = os.path.join(REPO_ROOT, "plots", "ceehrc_26")


def m6a_filter(fiber_idx, fiber_seq_inp):
    """fiber_seq_inp: (C, L, N) numpy array. Channel 0 is m6a (input_flags is
    all-active for these models, so no earlier feature is skipped)."""
    m6a_sum = (fiber_seq_inp[0, :, fiber_idx] > 0.5).sum()
    return m6a_sum > M6A_THRESHOLD


def move_dashboard(label, index):
    """Copies the existing composite dashboard PNG from ignore/ to plots/.
    Change shutil.copy2 -> shutil.move if you'd rather not keep the ignore/ copy."""
    src_dir = os.path.join(OUT_ROOT, label)
    matches = glob.glob(os.path.join(src_dir, f"{index:05d}_*.png"))
    if not matches:
        print(f"  [{label}] [WARN] no dashboard found for index {index} in {src_dir}")
        return
    src_path = matches[0]
    dst_dir = os.path.join(PLOTS_ROOT, label)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)
    print(f"  [{label}] copied {os.path.basename(src_path)} -> {dst_dir}")


def main():
    print(f"Repo root: {REPO_ROOT}")
    print(f"Selected indices: {SELECTED_INDICES}")

    filtered_ccres = build_filtered_ccres()
    models = load_models()

    ref_model = next(iter(models.values()))
    output_assays = ref_model.init_args.get("output_assays", ["atac"])
    input_flags = ref_model.init_args["input_flags"]
    dna_type = ref_model.init_args["dna_type"]

    dataset = build_dataset(output_assays, input_flags, dna_type)
    dataset.ccre_list = filtered_ccres
    dataset.num_sample_ccres = -1

    loader = DataLoader(dataset, batch_size=1, drop_last=False)

    selected_set = set(SELECTED_INDICES)
    remaining = set(SELECTED_INDICES)
    max_needed = max(SELECTED_INDICES) if SELECTED_INDICES else -1

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > max_needed:
                break
            if i not in selected_set:
                continue
            remaining.discard(i)

            chrom0 = batch["locus"][0][0]
            start0 = int(batch["locus"][1][0])
            end0 = int(batch["locus"][2][0])
            print(f"\n=== index {i}: {chrom0}:{start0}-{end0} ===", flush=True)

            fiber_features, target_bulk, forward_kwargs = unpack_batch(batch, DEVICE)

            for label, model in models.items():
                move_dashboard(label, i)

                with torch.amp.autocast(DEVICE_TYPE, dtype=torch.float16):
                    pred_bulk, processed_fibers = model(fiber_features, **forward_kwargs)

                dst_dir = os.path.join(PLOTS_ROOT, label)
                os.makedirs(dst_dir, exist_ok=True)

                K = len(output_assays)
                fig, _ = plot_single_column_fiber_stack_t(
                    true_bulk_t=[target_bulk[:, k] for k in range(K)],
                    pred_bulk_t=[pred_bulk[:, k] for k in range(K)],
                    fiber_seq_inp_t=fiber_features,
                    processed_fibers_t=[processed_fibers[:, k] for k in range(K)],
                    input_flags=input_flags,
                    locus=(chrom0, start0, end0),
                    filter_fn=m6a_filter,
                    start_idx=0,
                    max_fibers=NUM_FIBERS,
                    bulk_name=output_assays,
                    mode="Test",
                )
                if fig is None:
                    print(f"  [{label}] [WARN] no fibers passed m6a > {M6A_THRESHOLD}")
                    continue
                out_name = f"{i:05d}_{chrom0}_{start0}-{end0}_single_molecule.png"
                fig.savefig(os.path.join(dst_dir, out_name), dpi=150)
                plt.close(fig)
                print(f"  [{label}] saved single-molecule 4-column plot -> {out_name}")

    if remaining:
        print(f"\n[WARN] indices not reached (out of range of the filtered set): {sorted(remaining)}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
