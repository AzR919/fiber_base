"""
Data processing file

"""

import os
import pyft
import pysam
import torch
import random
import pyBigWig
import contextlib
import numpy as np

# from w_redirect import stdout_redirected
from torch.utils.data import IterableDataset

class fiber_data_iterator(IterableDataset):

    def __init__(self, fiber_data_path, other_bw,
                 fibers_per_entry, context_length,
                 iters_per_epoch, fasta_path,
                 chr_sizes_file=None):

        self.fiber_bam = pyft.Fiberbam(fiber_data_path)
        self.other_bw = pyBigWig.open(other_bw)

        self.fibers_per_entry = fibers_per_entry
        self.context_length = context_length
        self.iters_per_epoch = iters_per_epoch

        self.load_fasta(fasta_path)
        self.load_genomic_coords(chr_sizes_file)

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
        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
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

    def generate_loci(self):

        random_chr = random.choice(list(self.chr_sizes.keys()))

        random_start = random.randint(0, self.chr_sizes[random_chr])
        random_end = random_start + self.context_length

        return random_chr, random_start, random_end

    def get_fiber_data(self, chrom, start, end):

        ML_THRESHOLD = 100

        fibers = []
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                possible_fibers = self.fiber_bam.fetch(chrom, start, end)

        # with stdout_redirected(to=os.devnull):
        #     possible_fibers = self.fiber_bam.fetch(chrom, start, end)

        for fiber in possible_fibers:
            data = np.zeros(end-start, dtype=np.float32)
            # Skip short or secondary reads
            if fiber.end - fiber.start < 1000:
                continue

            # Extract high-confidence m6A in reference coordinates
            m6a_ref = []
            for pos, ref_pos, ml in zip(fiber.m6a.starts, fiber.m6a.reference_starts, fiber.m6a.ml):
                if ref_pos is None: continue
                if start <= ref_pos < end and ml >= ML_THRESHOLD:
                    m6a_ref.append(ref_pos-start)

            if len(m6a_ref) == 0:
                continue

            data[m6a_ref] = 1
            fibers.append(data)
            if len(fibers)==self.fibers_per_entry: break

        if len(fibers)!=self.fibers_per_entry: return None

        fibers_tensor = torch.from_numpy(np.array(fibers))

        return fibers_tensor.T

    def get_other_bw_data(self, chrom, start, end):

        return  torch.from_numpy(np.array(self.other_bw.values(chrom, start, end))).to(torch.float32)

    def __iter__(self):

        for _ in range(self.iters_per_epoch):

            found_possible_locus = False

            while not found_possible_locus:

                random_locus = self.generate_loci()

                fiber_tensor = self.get_fiber_data(*random_locus)
                if fiber_tensor is None:continue
                else: found_possible_locus = True

                other_tensor = self.get_other_bw_data(*random_locus)
                dna = self.onehot_for_locus(random_locus)

            yield fiber_tensor, dna, other_tensor

def tester():

    pass

if __name__=="__main__":

    tester()
