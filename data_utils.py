"""
Data processing file

"""

import os
import pyft
import pysam
import torch
import numpy as np

from torch.utils.data import IterableDataset


class fiber_data_loader(IterableDataset):

    def __init__(self, data_path, iters_in_epoch):

        self.bam = pyft.Fiberbam(data_path)

        self.iters_in_epoch = iters_in_epoch

        self.load_fasta()

    def load_fasta(self):

        fasta_path = os.fspath(self.fasta_file)
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA not found: {fasta_path}")
        if not os.path.exists(fasta_path + ".fai"):
            pysam.faidx(fasta_path)               # build index if needed
        self.fasta = pysam.FastaFile(fasta_path)

    def get_DNA_sequence(self, chrom, start, end):
        """
        Retrieve the sequence for a given chromosome and coordinate range from a fasta file.

        """
        # Ensure coordinates are within the valid range
        if start < 0 or end <= start:
            raise ValueError("Invalid start or end position")

        return self.fasta.fetch(chrom, start, end)

    def dna_to_onehot(self, sequence):
        # Create a mapping from nucleotide to index
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':4}

        # Convert the sequence to indices
        indices = torch.tensor([mapping[nuc.upper()] for nuc in sequence], dtype=torch.long)

        # Create one-hot encoding
        one_hot = torch.nn.functional.one_hot(indices, num_classes=5)

        # Remove the fifth column which corresponds to 'N'
        one_hot = one_hot[:, :4]

        return one_hot

    def onehot_for_locus(self, locus):
        """
        Helper to fetch DNA and convert to one-hot for a given locus [chrom, start, end].
        Returns a tensor [context_length, 4].

        """
        chrom, start, end = locus[0], int(locus[1]), int(locus[2])
        seq = self.get_DNA_sequence(chrom, start, end)
        return self.dna_to_onehot(seq)

    def __iter__(self):
        return super().__iter__()

    pass