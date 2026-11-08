"""DEV4 and its input ablations (TODO_extended.md T10): short name -> training run name, input flags
[m6A, CpG, MSP, nuc, FIRE]. Shared by ablations.py, ctcf_footprints.py and xi_promoters.py."""

MODELS = {
    "DEV4":   "LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04",
    "noCpG":  "ABL_V1_data24_dev_loo_Panc1_200U_model19_uct_mid_8A_no_cpg_train04",
    "noFIRE": "ABL_V1_data24_dev_loo_Panc1_200U_model16_uct_mid_8A_no_fire_msp_train04",
    "noAcc":  "ABL_V1_data24_dev_loo_Panc1_200U_model20_uct_mid_8A_cpg_only_train04",
}
FLAGS = {"DEV4": [1, 1, 1, 1, 1], "noCpG": [1, 0, 1, 1, 1], "noFIRE": [1, 1, 1, 1, 0], "noAcc": [0, 1, 0, 0, 0]}


def molecule_dir(model, targets):
    """Newest molecule_predict output dir of a model for a target set (e.g. 'ctcf_sites_chr21')."""
    import glob
    import os
    dirs = glob.glob(f"results/*_molecules_{MODELS[model]}_{targets}")
    if not dirs:
        raise FileNotFoundError(f"no molecule_predict output for {model} on {targets}")
    return max(dirs, key=os.path.getmtime)
