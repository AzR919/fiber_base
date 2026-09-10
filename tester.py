"""
File for random testing.
Should not be in the final version
"""

import os
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import wandb
import pyft
import pysam
import pyBigWig
import matplotlib.pyplot as plt

from args import get_args
from data_utils import *
from models import *
from utils import *
from data_utils import *
from evaluator import *
from vis_utils import (
    plot_evaluation_dashboard_t,
    plot_single_column_fiber_stack_t,
    plot_dna_tensor_logo_t,
    FEATURE_NAMES,
    FEATURE_COLORS,
    UNMETHYLATED_COLOR,
)


#--------------------------------------------------------------------------------------------------
# Custom fiber filter functions

def my_m6a_filter(fiber_idx, fiber_seq_inp):
    """fiber_seq_inp: (C, L, N) numpy array. Channel 0 is m6a."""
    m6a_counts = (fiber_seq_inp[0, :, fiber_idx] > 0.5).sum()
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
    dataset = SingleCellFiberDataset(**data_config)
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

    fig_t = plot_evaluation_dashboard_t(
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
        fig_s, last_idx = plot_single_column_fiber_stack_t(
            true_bulk_t=target_bulk,
            pred_bulk_t=pred_bulk,
            fiber_seq_inp_t=fiber_features,
            processed_fibers_t=process_fibers,
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

    cell_samples = []
    failed_sampling = False

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
    test_ind = setup_batch(test_in, device)
    bulk_sig, process_fibers = model(test_ind["fiber_features"], **test_ind)

    fig_t = plot_evaluation_dashboard_t(
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

    pass

def tester_0():

    base_dir = "./ignore/"
    epoch = 20
    run_type = "sum"

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
