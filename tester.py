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
    tester_0()
    pass
