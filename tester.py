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

import os
import torch
import numpy as np

def tester_4():
    # Base configuration directories
    plot_save_base_dir = "./ignore/pres"
    data_root = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER"
    fasta_path = "/home/azr/projects/def-maxwl/azr/data/misc/hg38.fa"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define metadata relationships for Cell Types and Assays
    cell_metadata = {
        "GM": {
            "full_name": "GM12878",
            "cram": f"{data_root}/GM12878/GM12878-fire-v0.1-filtered.cram",
            "ccre": f"{data_root}/GM12878/gm12878_ccres.bed",
            "assays": {
                "atac": {
                    "bw": f"{data_root}/GM12878/ENCFF603BJO_ATAC_seq.bigWig",
                    "model_dir": "./results/26-07-13_T13-08-19_gm_atac_avg_n_fibers/"
                },
                "h3k4me3": {
                    "bw": f"{data_root}/GM12878/ENCFF287HAO_H3K4me3.bigWig",
                    "model_dir": "./results/26-07-14_T00-34-17_gm_h3k4me3_avg_n_fibers/"
                }
            }
        },
        "K5": {
            "full_name": "K562",
            "cram": f"{data_root}/K562/K562-fire-v0.1-filtered.cram",
            # Explicitly using GM ccres for K562 as requested
            "ccre": f"{data_root}/GM12878/gm12878_ccres.bed",
            "assays": {
                "atac": {
                    "bw": f"{data_root}/K562/ENCFF102ARJ_ATAC_seq.bigWig",
                    "model_dir": "./results/26-07-13_T13-42-37_k5_atac_avg_n_fibers/"
                },
                "h3k4me3": {
                    "bw": f"{data_root}/K562/ENCFF911JVK_H3K4me3.bigWig",
                    "model_dir": "./results/26-07-13_T21-19-51_k5_h3k4me3_avg_n_fibers/"
                }
            }
        }
    }

    cell_types = ["GM", "K5"]
    assays = ["atac", "h3k4me3"]

    # Scale up search matrix scope to screen 100 seeds
    seeds = list(range(1, 101))

    # =================================================================
    # OUTER LOOP: SCREEN THROUGH 100 SEEDS FOR STRONG TARGET REGIONS
    # =================================================================
    for seed in seeds:
        # =================================================================
        # INNER LOOPS: CELL LINES & TARGET MOLECULAR ASSAYS
        # =================================================================
        for cell in cell_types:
            for assay in assays:
                # Extract file architecture parameters
                cell_info = cell_metadata[cell]
                assay_info = cell_info["assays"][assay]
                model_path = os.path.join(assay_info["model_dir"], "Model_epoch_25.pt")

                if not os.path.exists(model_path):
                    continue

                # 1. Instantiate the dataset worker using the current seed to locate a candidate locus
                kwargs = {
                    "fiber_data_path": cell_info["cram"],
                    "other_bw": assay_info["bw"],
                    "fibers_per_entry": 200,
                    "context_length": 2048,
                    "iters_per_epoch": 1024,
                    "fasta_path": fasta_path,
                    "input_flags": [1, 1, 1, 1, 1],
                    "ccre_path": cell_info["ccre"],
                    "seed": seed
                }

                t_set = fiber_data_iterator(**kwargs)
                t_set.init_worker_resources()

                # Get the candidate genomic coordinates
                chrom, start, end = t_set.generate_ccre_loci(jitter_range=0)

                # Fetch target BigWig array data first to test activity strength
                g_output = t_set.get_other_bw_data(chrom, start, end)

                # Convert to numpy/tensor safely to find max value
                if isinstance(g_output, list):
                    bw_max = max([torch.max(go).item() if isinstance(go, torch.Tensor) else np.max(go) for go in g_output])
                else:
                    bw_max = torch.max(g_output).item() if isinstance(g_output, torch.Tensor) else np.max(g_output)

                # CRITICAL CUTOFF FILTER: Skip inference and plots if it doesn't cross threshold activity peak
                if bw_max <= 1.5:
                    continue

                # 2. Threshold crossed! Load the model using your exact required code block
                model, config = FiberDeep01ResConv1dBlock.load_model(
                    filepath=model_path,
                    map_location=device
                )
                model.to(device)
                model.eval()

                print(f" Found active region! Seed {seed} | {cell}-{assay} | Max BW Signal: {bw_max:.2f} at {chrom}:{start}-{end}")

                # Create the specific output directory for this cell type and assay (e.g., ./ignore/pres/GM12878/atac)
                out_dir = os.path.join(plot_save_base_dir, cell_info["full_name"], assay)
                os.makedirs(out_dir, exist_ok=True)

                # Fetch matching structural fiber tensors now that the region is validated
                fiber_tensor, dna_tensor, n_fibers = t_set.get_fiber_data(chrom, start, end)

                m_input = fiber_tensor.unsqueeze(0)
                m_n_fibers = torch.tensor(n_fibers)

                # Model Inference Execution Phase
                with torch.no_grad():
                    m_output, processed_fibers = model(m_input, n_fibers=m_n_fibers)

                # Set plot filename to just the seed number
                plot_filename = f"seed_{seed}"

                # Generate and save diagnostic target plots inside the cell/assay directory
                plot_sample_out_fibers_plt(
                    f"{out_dir}/{plot_filename}", m_input, kwargs["input_flags"], 5,
                    m_output, processed_fibers, [g_output], [[chrom], [start], [end]],
                    "oof", avg_loss="NaN", mode="Eval"
                )

                plot_single_fibers_plt(
                    f"{out_dir}/{plot_filename}_single", m_input, kwargs["input_flags"], 5,
                    m_output, processed_fibers, [g_output], [[chrom], [start], [end]],
                    "oof", avg_loss="NaN", mode="Eval"
                )

    print("\nLocus screening complete. High-signal plots saved to their respective directories.")

def tester_3():

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

def tester_2():

    fire_bw = pyBigWig.open("/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/trackHub-v0.1/bb/all.percent.accessible.bw")

    start_p = 28748500
    chrom, start, end = "chr22", start_p, start_p+2048

    bw_data = torch.from_numpy(np.array(fire_bw.values(chrom, start, end))).to(torch.float32)
    zeros = torch.from_numpy(np.zeros_like(bw_data))


    plot_sample("./ignore", [], [bw_data], [zeros], ([chrom], [start], [end]), "fire_bw")

    pass

def tester_1():

    fiber_bam = pyft.Fiberbam("/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/GM12878-fire-v0.1-filtered.cram")
    pysam_fiber = pysam.AlignmentFile("/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/GM12878-fire-v0.1-filtered.cram")
    atac_bw = pyBigWig.open("/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/ENCFF603BJO_ATAC_seq.bigWig")
    dnas_bw = pyBigWig.open("/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/ENCFF743ULW_DNase.bigWig")

    start_p = 43706195
    chrom, start, end = "chr21", start_p, start_p+2048
    bw = dnas_bw

    def get_fiber_data(chrom, start, end):

        ML_THRESHOLD = 100
        fibers = np.zeros((200, 2048), dtype=np.float32)

        with suppress_stdout_stderr():
            possible_fibers = fiber_bam.fetch(chrom, start, end)

        for i, fiber in enumerate(possible_fibers):
            if i == 200: break
            lone_fiber = np.zeros((2048), dtype=np.float32)
            # data = np.zeros(end-start, dtype=np.float32)
            # Skip short or secondary reads
            # if fiber.end - fiber.start < 1000:
            #     continue

            # Extract high-confidence m6A in reference coordinates
            # m6a_ref = []
            # for pos, ref_pos, ml in zip(fiber.m6a.starts, fiber.m6a.reference_starts, fiber.m6a.ml):
            #     if ref_pos is None: continue
            #     if start <= ref_pos < end and ml >= ML_THRESHOLD:
            #         m6a_ref.append(ref_pos-start)

            # if len(m6a_ref) == 0:
            #     continue

            # fibers[i,m6a_ref] = 1
        #     data[m6a_ref] = 1
        #     fibers.append(data)
        #     if len(fibers)==self.fibers_per_entry: break

            AQ_THRESHOLD = 200

            for ref_pos, len, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
                if ref_pos is None: continue
                if start <= ref_pos < end and aq >= AQ_THRESHOLD:
                    lone_fiber[ref_pos-start:ref_pos-start+len] = 1

            fibers[i] = lone_fiber
        fibers_tensor = torch.from_numpy(np.array(fibers))

        return fibers_tensor.T

    bw_data = torch.from_numpy(np.array(bw.values(chrom, start, end))).to(torch.float32)
    fibers = get_fiber_data(chrom, start, end).unsqueeze(0)
    zeros = torch.from_numpy(np.zeros_like(bw_data))

    plot_sample("./ignore", fibers, [zeros], [bw_data], ([chrom], [start], [end]), "atac_fire_score_reads_msp_plot")
    pass

def tester_0():

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="liblab",
        # Set the wandb project where this run will be logged.
        project="Fiber",
        name="test_run_2",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "none",
            "dataset": "none",
            "epochs": 10,
        },
    )

    args = get_args()

    in_t = torch.rand(2,32,4)

    # in_t = torch.load("./ignore/input.pt", map_location=torch.device('cpu'))
    # out_t = torch.load("./ignore/output.pt", map_location=torch.device('cpu'))
    # tar_t = torch.load("./ignore/target.pt", map_location=torch.device('cpu'))

    model = Simple_Add_CNN_Model(200)

    run.watch(model)

    mod_out = model(in_t, None)

    run.log({"random":mod_out})
    run.finish()

    print("All_Done")

if __name__=="__main__":
    # tester_0()
    # tester_1()
    # tester_2()
    # tester_3()
    tester_4()
    pass
