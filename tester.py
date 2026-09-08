"""
File for random testing.
Should not be in the final version
"""

import os
import wandb
import pyft
import pysam
import torch
import pyBigWig

from args import get_args

from data_utils import *
from models import *
from utils import *
from eval_dataset import *
from evaluator import *

import os
import torch
import numpy as np


import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import logomaker

import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------
# Single-Column Fiber Stack Visualization Utility

def _render_single_fiber_features(ax, fiber_inp, input_flags):
    """
    Helper to render discrete/continuous input feature channels for a single fiber on a single Axes.
    """
    active_indices = [j for j, flag in enumerate(input_flags) if flag]

    for k, j in enumerate(active_indices):
        feat_name = FEATURE_NAMES[j]
        color = FEATURE_COLORS[j]
        fiber_feat = fiber_inp[k].cpu().detach().numpy()

        if feat_name in ["m6a", "cpg"]:
            # 1. Unmethylated background (-1)
            unmeth_indices = np.where(fiber_feat < -0.5)[0]
            if len(unmeth_indices) > 0:
                ax.scatter(
                    unmeth_indices,
                    np.full_like(unmeth_indices, -k),
                    marker='|',
                    color=UNMETHYLATED_COLOR,
                    s=15,
                    alpha=0.35,
                    linewidths=0.6
                )
            # 2. Methylated sites (+1)
            meth_indices = np.where(fiber_feat > 0.5)[0]
            if len(meth_indices) > 0:
                ax.scatter(
                    meth_indices,
                    np.full_like(meth_indices, -k),
                    marker='|',
                    color=color,
                    s=25,
                    alpha=0.85,
                    linewidths=1.0
                )
        else:
            # Continuous interval features (MSP, NUC, etc.)
            masked = (fiber_feat > 0.5).astype(float)
            diff = np.diff(masked, prepend=0.0, append=0.0)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            for s, e in zip(starts, ends):
                if e > s:
                    ax.axhspan(
                        -k - 0.35,
                        -k + 0.35,
                        xmin=s / len(fiber_feat),
                        xmax=e / len(fiber_feat),
                        color=color,
                        alpha=0.5,
                        lw=0
                    )

    ax.set_ylim(-len(active_indices) + 0.5, 0.5)
    ax.set_yticks([])

def plot_single_column_fiber_stack(
    true_bulk,
    pred_bulk,
    fiber_seq_inp,
    processed_fibers,
    input_flags,
    locus=None,
    filter_fn=None,
    start_idx=0,
    max_fibers=20,
    bulk_name="Bulk Signal",
    mode="Test"
):
    """
    Plots bulk signal and individual fiber predictions stacked vertically in a single column.

    Layout:
        [Bulk Signal Comparison]
        [Fiber 1: Fiber-Seq Input + Processed Fiber Signal]
        [Fiber 2: Fiber-Seq Input + Processed Fiber Signal]
        ...

    Args:
        true_bulk (Tensor or np.ndarray): Ground truth bulk signal of shape [L] or [1, L].
        pred_bulk (Tensor or np.ndarray): Predicted bulk signal of shape [L] or [1, L].
        fiber_seq_inp (Tensor): Raw input features of shape [1, C, L, N] or [C, L, N].
        processed_fibers (Tensor): Model processed fibers of shape [1, L, N] or [L, N].
        input_flags (list of bool): Flags indicating active input features.
        locus (tuple, optional): Locus info tuple (chr_name, start, end).
        filter_fn (callable, optional): Custom function `filter_fn(fiber_idx, fiber_seq_inp)`
                                        returning True if the fiber should be included.
        start_idx (int): The starting index in N to search for valid fibers.
        max_fibers (int): Maximum number of fibers to display in the plot (default: 20).
        bulk_name (str): Title label for true bulk signal.
        mode (str): Evaluation mode label.

    Returns:
        tuple: (fig, last_used_idx)
            - fig (matplotlib.figure.Figure or None): Plotted figure object.
            - last_used_idx (int): The index in N of the last fiber processed/evaluated.
    """
    # 1. Standardize Inputs to Numpy Array & Tensor Dimensions
    if isinstance(true_bulk, torch.Tensor):
        true_bulk = true_bulk.squeeze().cpu().detach().numpy()
    if isinstance(pred_bulk, torch.Tensor):
        pred_bulk = pred_bulk.squeeze().cpu().detach().numpy()

    if fiber_seq_inp.dim() == 3:
        fiber_seq_inp = fiber_seq_inp.unsqueeze(0)  # Shape: [1, C, L, N]
    if processed_fibers.dim() == 2:
        processed_fibers = processed_fibers.unsqueeze(0)  # Shape: [1, L, N]

    total_fibers = fiber_seq_inp.shape[-1]
    seq_len = fiber_seq_inp.shape[2]

    # 2. Select Eligible Fibers
    selected_indices = []
    curr_idx = start_idx

    while curr_idx < total_fibers and len(selected_indices) < max_fibers:
        if filter_fn is None or filter_fn(curr_idx, fiber_seq_inp):
            selected_indices.append(curr_idx)
        curr_idx += 1

    last_used_idx = curr_idx - 1 if len(selected_indices) > 0 else start_idx

    if len(selected_indices) == 0:
        print(f"[Plotter Warning] No fibers satisfied criteria starting from index {start_idx}.")
        return None, last_used_idx

    num_selected = len(selected_indices)

    # 3. Figure Layout Setup
    # Height rule: Bulk plot occupies 1 unit height. Each fiber stack occupies 1 unit height.
    fig_height = 2.2 * (1 + num_selected)
    fig = plt.figure(figsize=(16, fig_height))

    # 1 Bulk Plot Row + `num_selected` Fiber Rows
    gs = gridspec.GridSpec(1 + num_selected, 1, figure=fig, hspace=0.45)

    # 4. Render Top Plot: Bulk Comparison
    ax_bulk = fig.add_subplot(gs[0, 0])
    ax_bulk.plot(true_bulk, color='dimgray', lw=1.5, label=bulk_name)
    ax_bulk.plot(pred_bulk, color='darkorange', lw=1.5, label='Predicted', alpha=0.85)
    ax_bulk.set_ylabel("Bulk Intensity", fontsize=10, fontweight='bold')
    ax_bulk.legend(loc='upper right', frameon=False)
    ax_bulk.set_xlim(0, seq_len)

    title_str = f"Bulk & Single Fiber Stack Overview ({mode})"
    if locus is not None:
        chr_name, s, e = locus[0][0] if isinstance(locus[0], list) else locus[0], locus[1], locus[2]
        title_str += f"\n{chr_name}:{s}-{e}"
    ax_bulk.set_title(title_str, fontsize=12, fontweight='bold', pad=10)

    # 5. Render Single Fiber Plots
    x_coords = np.arange(seq_len)
    global_max_y = float(processed_fibers[0, :, selected_indices].max().item())
    global_max_y = max(1.0, global_max_y)

    for i, fiber_idx in enumerate(selected_indices):
        # Create a sub-gridspec for the combined fiber plot (Fiber-seq input + Processed signal)
        fiber_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i + 1, 0], hspace=0.0, height_ratios=[1, 1.2])

        # Top half: Fiber-seq Input Features
        ax_inp = fig.add_subplot(fiber_gs[0, 0], sharex=ax_bulk)
        fiber_inp = fiber_seq_inp[0, :, :, fiber_idx]
        _render_single_fiber_features(ax_inp, fiber_inp, input_flags)
        ax_inp.set_title(f"Fiber #{fiber_idx}", fontsize=9, fontweight='bold', loc='left', pad=2)
        ax_inp.set_xticklabels([])

        # Bottom half: Processed Fiber Accessibility Signal
        ax_proc = fig.add_subplot(fiber_gs[1, 0], sharex=ax_bulk)
        proc_signal = processed_fibers[0, :, fiber_idx].cpu().detach().numpy()

        # Filled area plot matching dynamic range
        ax_proc.plot(x_coords, proc_signal, color='firebrick', lw=1.2)
        ax_proc.fill_between(x_coords, 0, proc_signal, color='crimson', alpha=0.45)

        ax_proc.set_ylim(0, global_max_y * 1.05)
        ax_proc.set_ylabel("Accessibility", fontsize=8, fontweight='bold')

        if i < num_selected - 1:
            ax_proc.set_xticklabels([])
        else:
            ax_proc.set_xlabel("Genomic Position (bp)", fontsize=10, fontweight='bold')

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.92)
    return fig, last_used_idx

# Custom filter function: Only select fibers with at least 15 m6A methylation calls
def my_m6a_filter(fiber_idx, fiber_seq_inp):
    # Assumes channel 0 is m6a
    m6a_counts = (fiber_seq_inp[0, 0, :, fiber_idx] > 0.5).sum().item()
    return m6a_counts >= 100

def setup_batch(batch, device):
    batch_deviced = {}
    for key in batch:
        try:
            batch_deviced[key] = batch[key].to(device)
        except:
            batch_deviced[key] = batch[key]

    return batch_deviced

def tester_2():

    save_dir = "./ignore/h3_fcc_k5_final"
    chrom= "chr21"
    start = 31735071
    png_name = f"test_3"
    start_idx_s = [0, 20, 40]
    cell_type = "K562"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = "./results/26-08-20_T01-53-45_model06_data05_mixed_h3k4me3_fcc_model06_uct_train01_eval05_mixed_h3k4me3_fcc_mixed/Model_epoch_50.pt"
    model = UNet03ConvTransformerWithDNA.load_model(model_dir, device)[0]
    model.eval()
    input_flags = model.init_args["input_flags"]

    data_config_path = f"./configs/data/data09_k5_h3k4me3_fcc.yaml"
    data_config = load_config_file(data_config_path)
    data_config["input_flags"] = input_flags
    data_config["dna_type"] = "ref"
    data_config["iters_per_epoch"] = -1
    dataset = fiber_data_iterator(**data_config)
    dataset.init_worker_resources()
    dataloader = DataLoader(dataset)

    end = start+data_config["context_length"]
    locus = (chrom, start, end)
    locus_p = ([chrom], [start], [end])
    fiber_data = dataset.get_fiber_data(0, chrom, start, end, min_overlap=50)
    fiber_features = fiber_data[0].unsqueeze(0)
    n_fibers = torch.tensor(fiber_data[2])
    ref_dna = dataset.onehot_for_locus(locus).T.unsqueeze(0)
    target_bulk = dataset.get_other_bw_data(0, chrom, start, end).unsqueeze(0)

    with torch.amp.autocast(device_type, dtype=torch.float16):
        pred_bulk, process_fibers = model(fiber_features, ref_dna, n_fibers)

    fig_t = plot_evaluation_dashboard(
                fiber_features,
                input_flags,
                pred_bulk,
                process_fibers,
                target_bulk,
                locus_p,
                [f"{cell_type}"],
                dataset.bulk_name,
                mode="Instance"
            )


    fig_t.savefig(f"{save_dir}/{png_name}.png")
    plt.close(fig_t)

    for start_idx in start_idx_s:
        fig_s, last_idx = plot_single_column_fiber_stack(
            true_bulk=target_bulk,
            pred_bulk=pred_bulk,
            fiber_seq_inp=fiber_features,
            processed_fibers=process_fibers,
            input_flags=input_flags,
            locus=locus,
            filter_fn=my_m6a_filter,
            start_idx=start_idx,
            max_fibers=10
        )
        fig_s.savefig(f"{save_dir}/{png_name}_singles_{start_idx}.png", dpi=300, bbox_inches="tight")
        print(last_idx)
        plt.close(fig_s)


    pass



def tester_1():

    save_dir = "./ignore/mixed_h3_final/"
    chrom= "chr21"
    start = 31735071
    png_name = f"test_3"
    selected_locus = (chrom, start, start+5000)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model_dir = "./results/26-08-20_T01-53-45_model06_data05_mixed_h3k4me3_fcc_model06_uct_train01_eval05_mixed_h3k4me3_fcc_mixed/Model_epoch_50.pt"
    model = UNet03ConvTransformerWithDNA.load_model(model_dir, device)[0]
    model.eval()
    input_flags = model.init_args["input_flags"]

    data_config_path = f"./configs/evals/eval05_mixed_h3k4me3_fcc.yaml"
    data_config = load_config_file(data_config_path)
    data_config["input_flags"] = input_flags
    data_config["dna_type"] = "ref"
    data_config["num_sample_ccres"] = -1
    dataset = MixedCellFiberDataset(**data_config)
    dataset.init_worker_resources()
    dataloader = DataLoader(dataset)

    evaluator = Evaluator(model, dataset, batch_size=1, num_plots_to_log=5, device=device, seed=919)

    # eval_results = evaluator.evaluate(save_path=save_dir)
    # test_log_dict = {"test_loss": eval_results["composite"]["loss"]}
    # with open('output_test.yaml', 'w') as file:
    #     yaml.dump(test_log_dict, file,)

    # # Select locus records to visualize (e.g., top N samples or first N samples)
    # locus_records = eval_results.get("locus_records", [])
    # wandb_image_list = []

    # for idx, record in enumerate(locus_records):

    #     # Generate the 2-column deconvolution plot
    #     fig = plot_evaluator_record(
    #         record=record,
    #         input_flags=input_flags,
    #         loss=eval_results["composite"]["loss"],
    #         ct_losses=eval_results["per_cell_type"],
    #         bulk_name=dataset.bulk_name,
    #         mode="Test"
    #     )

    #     # Extract locus info for clean WandB image captioning
    #     chr_name = record["locus"][0][0]
    #     start = record["locus"][1][0]
    #     end = record["locus"][2][0]
    #     num_locus = eval_results["num_locus"]
    #     caption = f"Locus {idx}/{num_locus}: {chr_name}:{start}-{end}"

    #     # Always close local figures to prevent memory leaks in Matplotlib
    #     plt.savefig(f"./test_{idx}.png")
    #     plt.close(fig)

    # # Log all dashboard figures under a dedicated gallery panel in WandB
    # test_log_dict["Evaluation/Deconvolution_Dashboards"] = wandb_image_list

    cell_samples = []

    for cell_idx, ct in enumerate(dataset.cell_type_names):
        sample = dataset._sample_single_cell_type(cell_idx, ct, selected_locus)
        if sample is None and dataset.fiber_counts_per_cell[ct] > 0:
            failed_sampling = True
            break
        if sample is not None:
            cell_samples.append(sample)

    if failed_sampling or not cell_samples:
        raise NotImplementedError

    test_in = dataset._build_composite_sample(selected_locus, cell_samples)
    # for test_in in dataloader:
    test_ind = setup_batch(test_in, device)
    bulk_sig, process_fibers = model(test_ind["fiber_features"], **test_ind)

    fig_t = plot_evaluation_dashboard(
                test_ind["fiber_features"],
                input_flags,
                bulk_sig,
                process_fibers,
                test_ind["target_bulk"],
                test_in["locus"],
                test_in["cell_type"],
                dataset.bulk_name,
                avg_loss=0.0,
                mode="Eval"
            )


    fig_t.savefig(f"{save_dir}/test_0.png")
    plt.close(fig_t)


    #     fig = plot_evaluator_record(
    #                                 record=test_in,
    #                                 input_flags=input_flags,
    #                                 loss=0.0,
    #                                 ct_losses={"sd":0.0,"asd":0.0},
    #                                 bulk_name=dataset.bulk_name,
    #                                 mode="Test"
    #                             )

    #     plt.savefig("./test_1.png")
    #     plt.close(fig)
    #     pass

    pass

def plot_dna_tensor_logo(dna_tensor, region_slice=None, title="Fiber Consensus Sequence Logo"):
    """
    Plots a sequence logo from a DNA tensor of shape (4, L, N).

    Parameters:
    - dna_tensor: torch.Tensor or np.ndarray of shape (4, L, N)
    - region_slice: tuple/slice, optional sub-region (start, end) to zoom in
                    e.g., (2400, 2500) for a 100 bp window in the middle.
    - title: Title for the generated matplotlib plot.
    """
    if isinstance(dna_tensor, torch.Tensor):
        dna_tensor = dna_tensor.detach().cpu().numpy()

    # 1. Average across the fibers (N dimension) -> shape: (4, L)
    # This yields the nucleotide frequency/probability at each position
    ppm = dna_tensor.mean(axis=2)

    # 2. Transpose to shape (L, 4) for tabular format
    ppm_T = ppm.T  # Shape: (L, 4)

    # 3. Convert to Pandas DataFrame expected by logomaker
    # Standard channel order convention: [A, C, G, T]
    df = pd.DataFrame(ppm_T, columns=['A', 'C', 'G', 'T'])

    # 4. If sequence length L is large (e.g., 5000 bp), slice to a readable sub-window
    if region_slice is not None:
        start, end = region_slice
        df = df.iloc[start:end].reset_index(drop=True)

    # 5. Create the sequence logo
    fig, ax = plt.subplots(figsize=(14, 3))

    # logomaker handles color schemes and character heights automatically
    logo = logomaker.Logo(
        df,
        ax=ax,
        # color_scheme='dna', # Standard colors: A=Green, C=Blue, G=Yellow/Orange, T=Red
        vpad=0.05
    )

    # Style the plot
    logo.style_spines(visible=False)
    logo.style_spines(spines=['bottom'], visible=True)
    ax.set_ylabel("Probability / Frequency")
    ax.set_xlabel("Position (bp)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig("./test")
    return

def tester_0():

    base_dir = "./ignore/"
    epoch = 20
    # run_type = "avg"
    run_type = "sum"

    # output_name = f"{base_dir}output_{run_type}_{epoch}.npz"
    # pred_fibers_name = f"{base_dir}pred_fibers_{run_type}_{epoch}.npz"
    # target_name = f"{base_dir}target_{run_type}_{epoch}.npz"
    output_name = f"{base_dir}output_{epoch}.npz"
    pred_fibers_name = f"{base_dir}pred_fibers_{epoch}.npz"
    target_name = f"{base_dir}target_{epoch}.npz"

    output_np = np.load(output_name)["arr_0"]
    pred_fibers_np = np.load(pred_fibers_name)["arr_0"]
    target_np = np.load(target_name)["arr_0"]

    pass


if __name__=="__main__":
    if len(sys.argv)>1:
        print("starting t1")
        tester_1()
    else:
        tester_1()
    pass
