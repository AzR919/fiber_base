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
    # mixture realism test, GM12878 + HepG2, DEV4NF's channels (TODO_extended.md section 2b, first run, 23 Nov)
    "mixR0":  "MIXR0_NF_data27_mix_GM12878_HepG2_200U_model16_uct_mid_8A_no_fire_msp_train04",
    "mixR1a": "MIXR1A_NF_data36_mix_bal_GM12878_HepG2_200U_model16_uct_mid_8A_no_fire_msp_train04",
    "pairNF": "PAIR_NF_data28_GM12878_HepG2_200U_model16_uct_mid_8A_no_fire_msp_train04",
    # the same two on the second dev pair, K562 + H1 (seed runs, 23-24 Nov)
    "mixR1aB": "MIXR1A_NF_data37_mix_bal_K562_H1_200U_model16_uct_mid_8A_no_fire_msp_train04",
    "pairNFB": "PAIR_NF_data34_K562_H1_200U_model16_uct_mid_8A_no_fire_msp_train04",
    # baseline ladder (TODO_extended.md T11, 25 Nov), R1a supervision: rung 1 rules, rung 2 MLP on hand-crafted
    # features, rung 4 DEV4NF's network + those features; A = GM12878 + HepG2, B = K562 + H1
    "rulesA": "MIXR1A_RULES_data36_mix_bal_GM12878_HepG2_200U_model24_rules_train05",
    "rulesB": "MIXR1A_RULES_data37_mix_bal_K562_H1_200U_model24_rules_train05",
    "mlpA": "MIXR1A_MLP_data36_mix_bal_GM12878_HepG2_200U_model22_mlp_handcrafted_train04",
    "mlpB": "MIXR1A_MLP_data37_mix_bal_K562_H1_200U_model22_mlp_handcrafted_train04",
    "hcA": "MIXR1A_HC_data36_mix_bal_GM12878_HepG2_200U_model23_uct_mid_8A_no_fire_msp_handcrafted_train04",
    "hcB": "MIXR1A_HC_data37_mix_bal_K562_H1_200U_model23_uct_mid_8A_no_fire_msp_handcrafted_train04",
    # sealed test (reports/sealed_test_protocol.md), HCT116 + MCF-7, trained 27 Nov
    "sR1a": "SEALED_R1A_data38_mix_bal_HCT116_MCF_7_200U_model16_uct_mid_8A_no_fire_msp_train04",
    "sPair": "SEALED_PAIR_data39_HCT116_MCF_7_200U_model16_uct_mid_8A_no_fire_msp_train04",
    "sMLP": "SEALED_MLP_data38_mix_bal_HCT116_MCF_7_200U_model22_mlp_handcrafted_train04",
    "sRules": "SEALED_RULES_data38_mix_bal_HCT116_MCF_7_200U_model24_rules_train05",
}
FLAGS = {"DEV4": [1, 1, 1, 1, 1], "noFIRE": [1, 1, 1, 1, 0], "m6aCpG": [1, 1, 0, 0, 0], "noAcc": [0, 1, 0, 0, 0],
         "noCpG": [1, 0, 1, 1, 1], "mixR0": [1, 1, 1, 1, 0], "mixR1a": [1, 1, 1, 1, 0], "pairNF": [1, 1, 1, 1, 0],
         "mixR1aB": [1, 1, 1, 1, 0], "pairNFB": [1, 1, 1, 1, 0],
         **{k: [1, 1, 1, 1, 0] for k in ["rulesA", "rulesB", "mlpA", "mlpB", "hcA", "hcB",
                                         "sR1a", "sPair", "sMLP", "sRules"]}}
NAMES = {"DEV4": "DEV4 (all inputs)", "noFIRE": "no FIRE", "m6aCpG": "m6A + CpG", "noAcc": "no accessibility",
         "noCpG": "no CpG", "mixR0": "R0 mixture (GM12878 + HepG2)", "mixR1a": "R1a mixture (GM12878 + HepG2)",
         "pairNF": "pair (GM12878 + HepG2, own tracks)", "mixR1aB": "R1a mixture (K562 + H1)",
         "pairNFB": "pair (K562 + H1, own tracks)",
         "rulesA": "rules (GM12878 + HepG2)", "rulesB": "rules (K562 + H1)", "mlpA": "MLP on features (GM12878 + HepG2)",
         "mlpB": "MLP on features (K562 + H1)", "hcA": "network + features (GM12878 + HepG2)",
         "hcB": "network + features (K562 + H1)",
         "sR1a": "R1a-S (HCT116 + MCF-7 mixtures)", "sPair": "Pair-S (HCT116 + MCF-7, own tracks)",
         "sMLP": "MLP-S (HCT116 + MCF-7 mixtures)", "sRules": "Rules-S (HCT116 + MCF-7 mixtures)"}


# Training seeds: 919 (the default, runs above) and 2 (second seed, 11 Nov; slurm_batch_command.sh SEED=2 puts
# _seed2 after the run prefix).
SEEDS = [919, 2]
# every training seed in use (the R1a / pair seed runs of 23-24 Nov add 3); seeds.py keeps SEEDS for the ablations
TRAIN_SEEDS = [919, 2, 3]


def run_name(model, seed=919):
    """Training run name of a model for a training seed."""
    if seed == 919:
        return MODELS[model]
    prefix, rest = MODELS[model].split("_data", 1)
    return f"{prefix}_seed{seed}_data{rest}"


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
