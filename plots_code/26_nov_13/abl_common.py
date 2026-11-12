"""DEV4 and its input ablations (TODO_extended.md T10): short name -> training run name, input flags
[m6A, CpG, MSP, nuc, FIRE]. Shared by ablations.py, ctcf_footprints.py and xi_promoters.py."""

# Order = the order in tables and figures: inputs removed step by step (FIRE, then all fibertools calls, then m6A
# too), then CpG removed on its own.
MODELS = {
    "DEV4":   "LOO_Panc1_V1_data24_dev_loo_Panc1_200U_model15_uct_mid_8A_train04",
    "noFIRE": "ABL_V1_data24_dev_loo_Panc1_200U_model16_uct_mid_8A_no_fire_msp_train04",
    "m6aCpG": "ABL_V1_data24_dev_loo_Panc1_200U_model21_uct_mid_8A_m6a_cpg_train04",  # no fibertools calls
    "noAcc":  "ABL_V1_data24_dev_loo_Panc1_200U_model20_uct_mid_8A_cpg_only_train04",
    "noCpG":  "ABL_V1_data24_dev_loo_Panc1_200U_model19_uct_mid_8A_no_cpg_train04",
}
FLAGS = {"DEV4": [1, 1, 1, 1, 1], "noFIRE": [1, 1, 1, 1, 0], "m6aCpG": [1, 1, 0, 0, 0], "noAcc": [0, 1, 0, 0, 0],
         "noCpG": [1, 0, 1, 1, 1]}
NAMES = {"DEV4": "DEV4 (all inputs)", "noFIRE": "no FIRE", "m6aCpG": "m6A + CpG", "noAcc": "no accessibility",
         "noCpG": "no CpG"}


# Training seeds: 919 (the default, runs above) and 2 (second seed, 11 Nov; slurm_batch_command.sh SEED=2 puts
# _seed2 after the run prefix).
SEEDS = [919, 2]


def run_name(model, seed=919):
    """Training run name of a model for a training seed."""
    if seed == 919:
        return MODELS[model]
    prefix, rest = MODELS[model].split("_data24", 1)
    return f"{prefix}_seed{seed}_data24{rest}"


def tag(model, seed=919):
    """Output-file suffix: the model key, plus _seed<n> for other seeds (seed-919 files keep their old names)."""
    return model if seed == 919 else f"{model}_seed{seed}"


def molecule_dir(model, targets, seed=919):
    """Newest molecule_predict output dir of a model for a target set (e.g. 'ctcf_sites_chr21')."""
    import glob
    import os
    dirs = glob.glob(f"results/*_molecules_{run_name(model, seed)}_{targets}")
    if not dirs:
        raise FileNotFoundError(f"no molecule_predict output for {model} (seed {seed}) on {targets}")
    return max(dirs, key=os.path.getmtime)
