

import os
import sys

import pyft
import tqdm
import h5py # type: ignore
import pysam
import random
import pyBigWig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.special import expit

from utils import *


cram_path = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/GM12878-fire-v0.1-filtered.cram"
bw_path = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/ENCFF743ULW_DNase.bigWig"

cram_file = pysam.AlignmentFile(cram_path)
pyft_file = pyft.Fiberbam(cram_path)
bw_file = pyBigWig.open(bw_path)

ccre_path="/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/gm12878_ccres.bed"

n_fiber_mu, n_fiber_sig = 144.83937049410937, 50.78823172018935
fiber_starts_mu, fiber_start_sig = 8181.968619700219, 6872.420787351789
fiber_len_mu, fiber_len_sig = 18446.405152837528, 6076.954917272429

h5_base_path = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/Synth"

def create_fiber(length, base_prob=0.005, msp_threshold=5, max_msp_len=150):
    fiber_synth = np.zeros(length)
    running_len = 0

    # We define the "peak" of our MSP probability
    peak_prob = 0.8

    for i in range(length):
        # 1. Calculate the dynamic probability
        if running_len == 0:
            current_p = base_prob
        else:
            # Sigmoid growth: as running_len increases, prob climbs to peak_prob
            # We center the shift at msp_threshold
            growth = expit(running_len - msp_threshold)
            current_p = base_prob + (peak_prob - base_prob) * growth

            # Drop-off logic: If it's getting too long, kill the probability
            if running_len > max_msp_len * 0.8:
                current_p *= (1 - (running_len / max_msp_len))

        # 2. Roll the dice
        if np.random.rand() < current_p:
            fiber_synth[i] = 1
            running_len += 1
        else:
            fiber_synth[i] = 0
            running_len = 0

        # Hard reset if we exceed max length
        if running_len >= max_msp_len:
            running_len = 0

    return fiber_synth

def load_ccres():
    # Load BED file (columns: chrom, start, end, name, score, strand, type...)
    df = pd.read_csv(ccre_path, sep='\t', header=None, usecols=[0, 1, 2])
    df.columns = ['chrom', 'start', 'end']
    main_chrs = ["chr21"]
    possible_chr_sizes = bw_file.chroms()
    chr_sizes = {k: possible_chr_sizes[k] for k in main_chrs if k in possible_chr_sizes}
    # Filter for chromosomes present in your chr_sizes to avoid errors
    return df[df['chrom'].isin(chr_sizes.keys())].values

def generate_ccre_loci(ccre_chrom, ccre_start, ccre_end, jitter_range=0, context_length=2048):
    """
    Generates a genomic window centered around a random cCRE with optional jitter.

    @args:
        jitter_range (int): The maximum number of base pairs to shift the center.
                            e.g., 200 means a shift between -200 and +200 bp.
    """

    # 2. Calculate the "true" center of the cCRE
    true_center = (ccre_start + ccre_end) // 2

    # 3. Apply Jitter
    # This shifts the focus point slightly so the cCRE isn't always perfectly centered
    jitter = random.randint(-jitter_range, jitter_range)
    focal_point = true_center + jitter

    # 4. Create the window around the focal point
    half_window = context_length // 2
    random_start = focal_point - half_window
    random_end = random_start + context_length

    if random_start < 0:
        random_start = 0
        random_end = context_length

    return ccre_chrom, int(random_start), int(random_end)

def h5_creator(h5_path):

    ccre_list = load_ccres()
    new_samples_n_fiber = np.round(stats.norm.rvs(loc=n_fiber_mu, scale=n_fiber_sig, size=len(ccre_list))).astype(int)

    h5_file = h5py.File(h5_path, 'w')

    for i, (n_fiber, ccre) in tqdm.tqdm(enumerate(zip(new_samples_n_fiber, ccre_list))):

        if i > 5: break

        chrom, start, end = ccre

        ccre_name = f"{chrom}:{start}-{end}"

        fiber_tensor = np.array([create_fiber(5000) for _ in range(n_fiber)])

        h5_file.create_dataset(ccre_name, data=fiber_tensor)

    pass

def locus_to_str(chrom, start, end):
    """Converts coordinates to 'chrom:start-end' format."""
    return f"{chrom}:{start}-{end}"

def str_to_locus(locus_str):
    """
    Parses 'chrom:start-end' back into (chrom, start, end).
    Handles commas often found in manual inputs.
    """
    # Split chrom from the rest
    chrom, coords = locus_str.replace(',', '').split(':')
    # Split start and end
    start, end = coords.split('-')

    return chrom, int(start), int(end)

def get_m6a(fiber, start, end, Q_THRESHOLD=200, context_length=2048):

    m6a_data = np.zeros((context_length), dtype=np.float32)

    # for ref_pos, aq in zip(fiber.m6a.reference_starts, fiber.m6a.ml):
    #     if ref_pos is None: continue
    #     if start <= ref_pos < end and aq >= Q_THRESHOLD:
    #         m6a_data[ref_pos-start:ref_pos-start+len] = 1

    # 1. Convert lists to numpy arrays
    ref_starts = np.array(fiber.m6a.reference_starts, dtype=np.float32)
    qualities = np.array(fiber.m6a.ml, dtype=np.float32)

    # 2. Create a boolean mask for everything that passes the filters
    # - Within the genomic window
    # - Above the quality threshold
    # - Not None (numpy handles this well if converted correctly)
    mask = (ref_starts >= start) & (ref_starts < end) & (qualities >= Q_THRESHOLD)

    # 3. Extract the passing positions and calculate their relative offsets
    valid_positions = (ref_starts[mask] - start).astype(np.int32)

    # 4. Use "Fancy Indexing" to set all 1s at once
    m6a_data[valid_positions] = 1

    return m6a_data

def plot_dual_fiber_comparison(set1_data, set2_data, locus, extra, dir="./ignore"):
    """
    set1_data/set2_data: Tuples of (signals, fibers)
        - signals: List of 1D tensors to plot on top (e.g., [target, pred])
        - fibers: The (C, L, N) input tensor (we use channel 0 for m6a)
    locus: (chrom, start, end) identifiers
    """
    chr_name, start, end = locus
    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"Comparison_{chr_name}_{start}_{extra}.png")

    # Create 2x2 grid
    # Width ratios are equal, height ratios prioritize the fibers
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex='col', sharey='row',
                             gridspec_kw={'height_ratios': [1, 4]})

    (ax_sig1, ax_sig2), (ax_fib1, ax_fib2) = axes

    def plot_column(ax_sig, ax_fib, signals, fibers, title):
        # 1. Plot all 1D signals on the top axis
        colors = ['slategrey', 'black', 'orange', 'steelblue', 'crimson']
        for i, (sig, label) in enumerate(signals):
            ax_sig.plot(sig, color=colors[i % len(colors)], linestyle='-',
                    linewidth=1.5,alpha=0.8, label=label)
        ax_sig.set_title(title)
        ax_sig.legend(loc='upper right', fontsize='small')

        # 2. Plot m6a fibers (Channel 0)
        # fibers shape: (C, L, N). Channel 0 is m6a.
        m6a_matrix = fibers#[0].cpu().detach() # (L, N)
        num_fibers = m6a_matrix.shape[0]
        L = m6a_matrix.shape[0]

        for i in range(num_fibers):
            fiber = m6a_matrix[i, :]
            masked = (fiber > 0.5)

            # Vectorized finding of m6a points
            indices = np.where(masked == 1)[0]
            if len(indices) > 0:
                # Plot m6a as small ticks/points rather than spans for clarity
                ax_fib.scatter(indices, np.full_like(indices, -i),
                               marker='|', color='black', s=10, alpha=0.5)

        ax_fib.set_ylim(-num_fibers - 0.5, 0.5)
        ax_fib.set_xlabel("Genomic Position")

    # Execute plotting for both columns
    plot_column(ax_sig1, ax_fib1, set1_data[0], set1_data[1], "Synthetic")
    plot_column(ax_sig2, ax_fib2, set2_data[0], set2_data[1], "Fiber_seq")

    ax_sig1.set_ylabel("Signal Intensity")
    ax_fib1.set_ylabel("Individual Fibers")

    plt.suptitle(f"Locus Comparison: {chr_name}:{start}-{end}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def norm_min_max(sig):

    sig_max = sig.max()
    sig_min = sig.min()

    normed = (sig-sig_min) / (sig_max-sig_min)

    return normed, sig_max, sig_min

def plotter(h5_path, extra):

    h5_file = h5py.File(h5_path, 'r')

    ccres = h5_file.keys()

    window_size = 2048

    for ccre in ccres:
        locus = str_to_locus(ccre)
        locus = generate_ccre_loci(*locus)
        synth_tensor = h5_file[ccre][:]

        start_idx = (synth_tensor.shape[1] - window_size) // 2
        synth_middle_window = synth_tensor[:, start_idx : start_idx + window_size]

        synth_sum, synth_sum_max, synth_sum_min = norm_min_max(synth_middle_window.sum(axis=0))
        synth_sum_title = f"Sum:[{synth_sum_min:.3f}, {synth_sum_max:.3f}]"

        with suppress_stdout_stderr():
            possible_fibers = pyft_file.fetch(*locus)

        fiber_tensor = np.array([get_m6a(fiber, locus[1], locus[2]) for fiber in possible_fibers])
        fiber_sum, fiber_sum_max, fiber_sum_min = norm_min_max(fiber_tensor.sum(axis=0))
        fiber_sum_title = f"Sum:[{fiber_sum_min:.3f}, {fiber_sum_max:.3f}]"

        dnase_sig, dnase_max, dnase_min = norm_min_max(np.array(bw_file.values(*locus)))
        dnase_title = f"Dnase:[{dnase_min:.3f}, {dnase_max:.3f}]"

        plot_dual_fiber_comparison([[(synth_sum, synth_sum_title)],synth_middle_window], [[(fiber_sum, fiber_sum_title), (dnase_sig, dnase_title)],fiber_tensor], locus, extra)

    pass


def main():
    extra = "test_0"
    h5_path = f"{h5_base_path}/synth_{extra}.h5"
    h5_creator(h5_path)
    plotter(h5_path, extra)
    pass

if __name__=="__main__":
    main()
