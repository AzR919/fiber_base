

import os
import sys

import pyft
import pysam
import pyBigWig

import matplotlib.pyplot as plt
import numpy as np

cram_path = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/GM12878-fire-v0.1-filtered.cram"
bw_path = "/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/ENCFF743ULW.bw"

cram_file = pysam.AlignmentFile(cram_path)
pyft_file = pyft.Fiberbam(cram_path)
bw_file = pyBigWig.open(bw_path)

def try_1():

    # Inputs (customize these)
    bam_file = cram_path  # Fiber-seq BAM from fibertools
    bigwig_file = bw_path  # DNase-seq signal
    region = 'chr11:119084800-119085800'  # Example locus (HMBS)
    chrom, start, end = region.replace(':', '-').split('-')  # Parse region
    start, end = int(start), int(end)
    fiber_height = 0.5  # Height per fiber track
    num_fibers_to_plot = 50  # Limit for visualization (papers show ~50-100)

    # Step 1: Load DNase-seq signal
    bw = pyBigWig.open(bigwig_file)
    dnase_positions = np.arange(start, end)
    dnase_values = bw.values(chrom, start, end)
    bw.close()

    # Step 2: Fetch Fiber-seq reads and m6A positions
    samfile = pysam.AlignmentFile(bam_file, 'rb')
    fibers = []  # List of (read_name, read_start, read_end, m6a_positions)
    for read in pyft_file.fetch(chrom, start, end):
        if read.is_secondary or read.is_supplementary:
            continue
        read_start, read_end = read.reference_start, read.reference_end
        
        # Extract m6A positions (from MM/ML tags; see PacBio BAM spec)
        m6a_positions = []
        if 'MM' in read.tags and 'ML' in read.tags:
            mm = read.get_tag('MM').split(';')[0]  # e.g., 'Z' for m6A
            ml = read.get_tag('ML')
            if mm.startswith('Z'):  # m6A modifications
                offsets = [int(o) for o in mm[1:].split(',')]  # Offsets from read start
                for i, offset in enumerate(offsets):
                    if ml[i] > 200:  # Threshold for high-confidence m6A (adjust based on papers, e.g., >90% prob)
                        m6a_positions.append(read_start + offset)
        
        fibers.append((read.query_name, read_start, read_end, m6a_positions))

    samfile.close()

    # Sort fibers (e.g., by number of m6A, as in fibertools Fig 1G)
    fibers.sort(key=lambda x: len(x[3]), reverse=True)  # Or sort by len(x[2] - x[1]) for length
    fibers = fibers[:num_fibers_to_plot]  # Limit

    # Step 3: Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 4]})

    # Top: DNase-seq
    ax1.fill_between(dnase_positions, dnase_values, color='blue', alpha=0.5)
    ax1.set_ylabel('DNase-seq Signal')
    ax1.set_title(f'DNase-seq and Fiber-seq at {region}')

    # Bottom: Fibers
    for i, (name, r_start, r_end, m6a) in enumerate(fibers):
        y_pos = -i * fiber_height  # Stack downward
        ax2.plot([r_start, r_end], [y_pos, y_pos], color='black', linewidth=0.5)  # Fiber line
        ax2.scatter(m6a, [y_pos] * len(m6a), color='red', s=10, marker='o')  # m6A dots

    ax2.set_ylabel('Individual Fibers')
    ax2.set_xlabel('Genomic Position')
    ax2.set_xlim(start, end)
    ax2.set_ylim(-len(fibers) * fiber_height - 1, 1)  # Adjust for stacking

    plt.tight_layout()
    plt.savefig('fiberseq_plot.png', dpi=300)
    plt.show()
    pass

def try_2():

    # ================== CONFIG ==================
    BAM_FILE = cram_path        # Must be processed with `ft call-m6a`
    BIGWIG_FILE = bw_path           # ENCODE DNase-seq
    REGION = "chr11:119084800-119085800"      # HMBS locus (fibertools Fig 1G)
    MAX_FIBERS = 50
    FIBER_HEIGHT = 0.4
    ML_THRESHOLD = 200  # High-confidence m6A (0–255 scale; ~80%+ probability)

    # Parse region
    chrom, coords = REGION.split(":")
    start, end = map(int, coords.split("-"))

    # ================== 1. Load DNase-seq ==================
    bw = pyBigWig.open(BIGWIG_FILE)
    x = np.arange(start, end)
    dnase_signal = np.nan_to_num(bw.values(chrom, start, end), nan=0.0)
    bw.close()

    # ================== 2. Extract Fiber-seq with pyft ==================
    fb = pyft.Fiberbam(BAM_FILE)  # Opens BAM (must be indexed)

    fibers = []
    for fiber in fb.fetch(REGION):  # ← This is the correct way
        # Skip short or secondary reads
        if fiber.end - fiber.start < 1000:
            continue

        # Extract high-confidence m6A in reference coordinates
        m6a_ref = []
        for ref_pos, ml in zip(fiber.m6a.reference_starts, fiber.m6a.ml):
            if ref_pos is None: continue
            if start <= ref_pos <= end and ml >= ML_THRESHOLD:
                m6a_ref.append(ref_pos)
        
        if len(m6a_ref) == 0:
            continue

        fibers.append({
            'name': fiber.qname,
            'start': fiber.start,
            'end': fiber.end,
            'm6a': m6a_ref,
            'ml_scores': fiber.m6a.ml,
            'ccs_passes': fiber.ec  # Number of CCS passes
        }) # Always close

    # Sort by number of m6A (descending) → matches paper
    fibers.sort(key=lambda x: len(x['m6a']), reverse=True)
    fibers = fibers[:MAX_FIBERS]

    print(f"Plotted {len(fibers)} fibers with ≥{ML_THRESHOLD} ML in {REGION}")

    # ================== 3. Plot ==================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                gridspec_kw={'height_ratios': [1, 4]})

    # Top: DNase-seq
    ax1.fill_between(x, dnase_signal, color='steelblue', alpha=0.7, label='DNase-seq')
    ax1.set_ylabel("DNase Signal")
    ax1.set_title(f"Fiber-seq at {REGION} (HMBS)")
    ax1.legend()

    # Bottom: Fibers
    for i, f in enumerate(fibers):
        y = -i * FIBER_HEIGHT
        # Fiber backbone
        ax2.hlines(y, f['start'], f['end'], color='black', lw=0.6)
        # m6A dots (red)
        ax2.scatter(f['m6a'], [y] * len(f['m6a']), color='red', s=15, zorder=5)

    ax2.set_ylim(-len(fibers) * FIBER_HEIGHT - 0.5, 0.5)
    ax2.set_ylabel("Individual Fibers\n(sorted by # m6A)")
    ax2.set_xlabel("Genomic Position (bp)")
    ax2.set_xlim(start, end)

    plt.tight_layout()
    plt.savefig("fiberseq_plot_final.png", dpi=300, bbox_inches='tight')
    plt.show()
    pass


def main():
    try_2()
    pass

if __name__=="__main__":
    main()
