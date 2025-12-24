"""
Main file

"""

import os
import sys
import datetime

from args import get_args
from data_utils import fiber_data_iterator
from trainer import Trainer
from models import *
from utils import *

def create_save_str(args):

    now = datetime.datetime.now()
    now = now.strftime("%y-%m-%d_T%H-%M-%S")

    save_str = f"{now}_{args.name_suffix}"

    return save_str

def model_selector(model_arg, args):

    model_name = model_arg.lower()

    if model_name=="base": return Base_Model(args.fibers_per_entry)
    if model_name=="simple": return Simple_Add_CNN_Model(args.fibers_per_entry)

    raise NotImplementedError(f"Model not implemented: {model_arg}")

def main():

    args = get_args()
    set_seed(args.seed)

    save_str = create_save_str(args)
    res_dir = os.path.join(args.res_dir, save_str)

    data_iterator = fiber_data_iterator(args.fiber_data_path, args.other_data_path,
            fibers_per_entry=args.fibers_per_entry, context_length=args.context_length,
            iters_per_epoch=args.iters_per_epoch, fasta_path="/home/azr/misc/hg38.fa",
            ccre_path="/home/azr/projects/def-maxwl/azr/data/DATA_FIBER/GM12878/grch38_ccres.bed")

    model = model_selector(args.model, args)

    trainer = Trainer(model, data_iterator, epochs=args.epochs, batch_size=args.batch_size)

    trainer.train(save_dir=res_dir)

if __name__=="__main__":
    main()
