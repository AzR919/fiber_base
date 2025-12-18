"""
Main file

"""

import os
import sys


from args import get_args
from data_utils import fiber_data_iterator
from models import Base_Model
from trainer import Trainer
from utils import *


def main():

    args = get_args()
    set_seed(args.seed)

    data_iterator = fiber_data_iterator(args.fiber_data_path, args.other_data_path,
            fibers_per_entry=args.fibers_per_entry, context_length=args.context_length,
            iters_per_epoch=args.iters_per_epoch, fasta_path="/home/azr/misc/hg38.fa")

    model = Base_Model(args.fibers_per_entry)

    trainer = Trainer(model, data_iterator, epochs=args.epochs, batch_size=args.batch_size)

    trainer.train()

if __name__=="__main__":
    main()
