"""
File for random testing.
Should not be in the final version
"""

import pyft
import pysam
import torch
import pyBigWig

from args import get_args

from models import Base_Model, Simple_Add_CNN_Model

from utils import *


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

    args = get_args()

    in_t = torch.load("./ignore/input.pt", map_location=torch.device('cpu'))
    out_t = torch.load("./ignore/output.pt", map_location=torch.device('cpu'))
    tar_t = torch.load("./ignore/target.pt", map_location=torch.device('cpu'))

    model = Simple_Add_CNN_Model(200)

    mod_out = model(in_t, None)

    print("All_Done")

if __name__=="__main__":
    # tester_0()
    tester_1()
    # tester_2()
