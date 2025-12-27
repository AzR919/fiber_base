"""
File for random testing.
Should not be in the final version
"""

import torch

from args import get_args

from models import Base_Model, Simple_Add_CNN_Model


def main():

    args = get_args()

    in_t = torch.load("./ignore/input.pt", map_location=torch.device('cpu'))
    out_t = torch.load("./ignore/output.pt", map_location=torch.device('cpu'))
    tar_t = torch.load("./ignore/target.pt", map_location=torch.device('cpu'))

    model = Simple_Add_CNN_Model(200)

    mod_out = model(in_t, None)

    print("All_Done")

if __name__=="__main__":
    main()
