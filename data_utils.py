"""
Data processing file

"""

import os
import pyft
import pysam
import torch
import random
import pyBigWig
import numpy as np
import pandas as pd

# from w_redirect import stdout_redirected
from torch.utils.data import IterableDataset

from utils import *

class fiber_data_iterator(IterableDataset):

    def __init__(self, fiber_data_path, other_bw,
                 fibers_per_entry, context_length,
                 iters_per_epoch, fasta_path, ccre_path,
                 chr_sizes_file=None):

        self.fiber_bam = pyft.Fiberbam(fiber_data_path)
        self.other_bw = pyBigWig.open(other_bw)

        self.fibers_per_entry = fibers_per_entry
        self.context_length = context_length
        self.iters_per_epoch = iters_per_epoch

        self.load_fasta(fasta_path)
        self.load_genomic_coords(chr_sizes_file)

        self.load_ccres(ccre_path)

    def load_fasta(self, fasta_path):

        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")
        if not os.path.exists(fasta_path + ".fai"):
            pysam.faidx(fasta_path)               # build index if needed
        self.fasta = pysam.FastaFile(fasta_path)

    def onehot_for_locus(self, locus):
        """
        Helper to fetch DNA and convert to one-hot for a given locus [chrom, start, end].
        Returns a tensor [context_length, 4].

        """
        def get_DNA_sequence(chrom, start, end):
            """
            Retrieve the sequence for a given chromosome and coordinate range from a fasta file.

            """
            # Ensure coordinates are within the valid range
            if start < 0 or end <= start:
                raise ValueError("Invalid start or end position")

            return self.fasta.fetch(chrom, start, end)

        def dna_to_onehot(sequence):
            # Create a mapping from nucleotide to index
            mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}

            # Convert the sequence to indices
            indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)

            # Create one-hot encoding
            one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

            # Remove the fifth column which corresponds to 'N'
            one_hot = one_hot[:, :4]

            return one_hot.to(torch.float32)

        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        seq = get_DNA_sequence(chrom, start, end)
        return dna_to_onehot(seq)

    def load_genomic_coords(self, chr_sizes_file, mode="train"):
        # main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        main_chrs = ["chr21"]
        # if mode == "train":
        #     main_chrs.remove("chr21") # reserved for test
        # self.chr_sizes = {}

        # with open(chr_sizes_file, 'r') as f:
        #     for line in f:
        #         chr_name, chr_size = line.strip().split('\t')
        #         if chr_name in main_chrs:
        #             self.chr_sizes[chr_name] = int(chr_size)

        # self.genomesize = sum(list(self.chr_sizes.values()))
        possible_chr_sizes = self.other_bw.chroms()
        self.chr_sizes = {k: possible_chr_sizes[k] for k in main_chrs if k in possible_chr_sizes}

    def load_ccres(self, bed_path):
        # Load BED file (columns: chrom, start, end, name, score, strand, type...)
        df = pd.read_csv(bed_path, sep='\t', header=None, usecols=[0, 1, 2])
        df.columns = ['chrom', 'start', 'end']
        # Filter for chromosomes present in your chr_sizes to avoid errors
        self.ccre_list = df[df['chrom'].isin(self.chr_sizes.keys())].values

    def generate_loci(self):

        random_chr = random.choice(list(self.chr_sizes.keys()))

        random_start = random.randint(0, self.chr_sizes[random_chr])
        random_end = random_start + self.context_length

        return random_chr, random_start, random_end

    def generate_ccre_loci(self, jitter_range=200):
        """
        Generates a genomic window centered around a random cCRE with optional jitter.

        @args:
            jitter_range (int): The maximum number of base pairs to shift the center.
                                e.g., 200 means a shift between -200 and +200 bp.
        """
        # 1. Pick a random cCRE
        ccre_chrom, ccre_start, ccre_end = random.choice(self.ccre_list)

        # 2. Calculate the "true" center of the cCRE
        true_center = (ccre_start + ccre_end) // 2

        # 3. Apply Jitter
        # This shifts the focus point slightly so the cCRE isn't always perfectly centered
        jitter = random.randint(-jitter_range, jitter_range)
        focal_point = true_center + jitter

        # 4. Create the window around the focal point
        half_window = self.context_length // 2
        random_start = focal_point - half_window
        random_end = random_start + self.context_length

        # 5. Boundary Check (Crucial to prevent index errors)
        max_size = self.chr_sizes[ccre_chrom]

        if random_start < 0:
            random_start = 0
            random_end = self.context_length
        elif random_end > max_size:
            random_end = max_size
            random_start = max_size - self.context_length

        return ccre_chrom, int(random_start), int(random_end)

    def get_fiber_data(self, chrom, start, end):

        AQ_THRESHOLD = 200
        fibers = np.zeros((self.fibers_per_entry, self.context_length), dtype=np.float32)

        with suppress_stdout_stderr():
            possible_fibers = self.fiber_bam.fetch(chrom, start, end)

        for i, fiber in enumerate(possible_fibers):
            if i == self.fibers_per_entry: break

            data = np.zeros((2048), dtype=np.float32)

            for ref_pos, len, aq in zip(fiber.msp.reference_starts, fiber.msp.reference_lengths, fiber.msp.qual):
                if ref_pos is None: continue
                if start <= ref_pos < end and aq >= AQ_THRESHOLD:
                    data[ref_pos-start:ref_pos-start+len] = 1

            fibers[i] += data

            # data_nuc = np.zeros((2048), dtype=np.float32)

            # for ref_pos, len in zip(fiber.nuc.reference_starts, fiber.nuc.reference_lengths):
            #     if ref_pos is None: continue
            #     if start <= ref_pos < end:
            #         data_nuc[ref_pos-start:ref_pos-start+len] = -1

            # fibers[i] += data_nuc

        fibers_tensor = torch.from_numpy(np.array(fibers))

        return fibers_tensor.T

    def get_other_bw_data(self, chrom, start, end):

        return torch.asinh(torch.from_numpy(np.array(self.other_bw.values(chrom, start, end))).to(torch.float32))

    def __iter__(self):

        for _ in range(self.iters_per_epoch):

            found_possible_locus = False

            while not found_possible_locus:

                random_locus = self.generate_ccre_loci()

                fiber_tensor = self.get_fiber_data(*random_locus)
                if fiber_tensor is None : continue

                other_tensor = self.get_other_bw_data(*random_locus)
                has_nan = torch.isnan(other_tensor).any().item()
                if has_nan : continue

                dna = self.onehot_for_locus(random_locus)
                found_possible_locus = True

            yield fiber_tensor, dna, other_tensor, random_locus

#--------------------------------------------------------------------------------------------------
# testing

def tester():
    pass

if __name__=="__main__":

    tester()
