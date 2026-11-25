"""
Hand-crafted per-position features for the simple-model baselines (reports/TODO_extended.md T11, the ladder):
  rung 2  FeatureMLP: these features -> a small per-position MLP (model 'mlp01')
  rung 4  the network with these features as extra fiber input channels (unet03 with extra_features: true)

The features use only what the network sees: the fiber's own channels (m6A, CpG, MSP, nucleosome) and the reference
DNA. No external annotation (no CpG-island, TSS, gene or repeat track, no motif database), so rung 2 and rung 4 stay
comparable to the network (user, 25 Nov). They are fixed (no trainable parameters) and computed on the fly from the
model inputs, so the training pipeline, the evaluator and molecule_predict.py (including --perturb CpG flips) need no
change.

Input encoding (fiber_utils): m6A +1 = call, -1 = A/T without a call, 0 = no information; CpG: both bases of a covered
CpG -1, the base carrying a 5mC call +1 (so a methylated CpG has one +1, an unmethylated one two -1); MSP / nucleosome
1 inside, 0 outside.

MSPs include the short linkers between nucleosomes, so the rule parts use large MSPs (runs >= 151 bp: the
nucleosome-free regions) for "open region".

Features (N_FEATURES = 31), per fiber and position:
  fiber summaries, over windows of 51, 201 and 1001 bp centred on the position (5 x 3):
    MSP fraction, nucleosome fraction, m6A rate (calls / A-T positions with information), CpG methylation (methylated /
    covered CpGs), covered-CpG density (covered CpGs per bp)
  annotation-free parts of the rules (10):
    inside an MSP; inside a large MSP; a large MSP within 100 / 300 / 1000 / 2000 bp; a nucleosome within 300 / 1000 bp
    of a large MSP; a motif-free footprint (inside a large MSP, the 15 bp around the position have an m6A rate <= 0.5 x
    the 100 bp on either side) and a footprint within 100 bp
  reference sequence, over 201 and 1001 bp (3 x 2): GC fraction, CpG density, CpG observed / expected
"""

import torch
import torch.nn as nn

WINDOWS = (25, 100, 500)          # half widths: 51, 201, 1001 bp
MSP_REACH = (100, 300, 1000, 2000)
NUC_REACH = (300, 1000)
SEQ_WINDOWS = (100, 500)
FP_CORE, FP_FLANK = 7, 107        # footprint: 15 bp core, 100 bp flanks on each side
BIG_MSP_HALF = 75                 # large MSP: a run of >= 2 * 75 + 1 = 151 bp
N_FEATURES = 5 * len(WINDOWS) + 2 + len(MSP_REACH) + len(NUC_REACH) + 2 + 3 * len(SEQ_WINDOWS)
REQUIRED_FLAGS = [1, 1, 1, 1, 0]  # m6A, CpG, MSP, nucleosome (no FIRE), in this order


def boxsum(x, half):
    """Sum of x over [p - half, p + half] along the last dim (zero padded), via cumulative sums: O(L) whatever the
    window. x: (..., L) float."""
    L = x.shape[-1]
    c = torch.cumsum(torch.nn.functional.pad(x, (1, 0)), dim=-1)          # c[i] = sum(x[:i])
    idx = torch.arange(L, device=x.device)
    hi = (idx + half + 1).clamp(max=L)
    lo = (idx - half).clamp(min=0)
    return c[..., hi] - c[..., lo]


def within(mask, reach):
    """1 where mask has a 1 within `reach` bp (dilation)."""
    return (boxsum(mask, reach) > 0).float()


def large_runs(mask, half):
    """Positions of mask that belong to a run of >= 2 * half + 1 ones: the run's fully-covered window centres,
    dilated back by half."""
    full = (boxsum(mask, half) >= 2 * half + 1).float()
    return within(full, half) * mask


def ratio(num, den):
    return torch.where(den > 0, num / den.clamp(min=1e-6), torch.zeros_like(num))


class FiberFeatures(nn.Module):
    """Fixed features (no parameters) from the active fiber channels (m6A, CpG, MSP, nucleosome) and the reference
    DNA. forward(x [B, 4, L, N], ref_dna [B, 4, L]) -> [B, N_FEATURES, L, N] float32."""

    def forward(self, x, ref_dna):
        B, C, L, N = x.shape
        assert C == 4, f"FiberFeatures needs the channels m6A, CpG, MSP, nucleosome (got {C} channels)"
        with torch.autocast(device_type=x.device.type, enabled=False):
            f = x.float().permute(0, 3, 1, 2).reshape(B * N, C, L)          # [BN, 4, L]
            m6a, cpg, msp, nuc = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
            m6_call, m6_info = (m6a > 0).float(), (m6a != 0).float()
            cg_meth, cg_cov = (cpg > 0).float(), (cpg != 0).float() / 2     # per CpG: one +1; covered: 2 nonzero
            feats = []
            for h in WINDOWS:
                w = 2 * h + 1
                feats += [boxsum(msp, h) / w, boxsum(nuc, h) / w,
                          ratio(boxsum(m6_call, h), boxsum(m6_info, h)),
                          ratio(boxsum(cg_meth, h), boxsum(cg_cov, h)).clamp(max=1),
                          boxsum(cg_cov, h) / w]
            big = large_runs(msp, BIG_MSP_HALF)
            feats += [msp, big]
            feats += [within(big, r) for r in MSP_REACH]
            feats += [nuc * within(big, r) for r in NUC_REACH]
            core_call, core_info = boxsum(m6_call, FP_CORE), boxsum(m6_info, FP_CORE)
            fl_call, fl_info = boxsum(m6_call, FP_FLANK) - core_call, boxsum(m6_info, FP_FLANK) - core_info
            flank_rate = ratio(fl_call, fl_info)
            fp = ((core_info >= 4) & (flank_rate > 0) & (ratio(core_call, core_info) <= 0.5 * flank_rate)).float() * big
            feats += [fp, within(fp, 100)]
            fib = torch.stack(feats, dim=1)                                   # [BN, F_fib, L]

            d = ref_dna.float()                                               # [B, 4, L]: A, C, G, T
            gc = d[:, 1] + d[:, 2]
            cpg_site = torch.nn.functional.pad(d[:, 1, :-1] * d[:, 2, 1:], (0, 1))   # C at p and G at p+1
            seq = []
            for h in SEQ_WINDOWS:
                w = 2 * h + 1
                c_n, g_n = boxsum(d[:, 1], h), boxsum(d[:, 2], h)
                cg_n = boxsum(cpg_site, h)
                seq += [boxsum(gc, h) / w, cg_n / w, ratio(cg_n * w, c_n * g_n).clamp(max=3)]
            seq = torch.stack(seq, dim=1)                                     # [B, F_seq, L]
            seq = seq.unsqueeze(1).expand(B, N, -1, L).reshape(B * N, -1, L)
            out = torch.cat([fib, seq], dim=1)                                # [BN, N_FEATURES, L]
            return out.reshape(B, N, -1, L).permute(0, 2, 3, 1).contiguous()  # [B, F, L, N]


# --------------------------------------------------------------------------------------------------------------------
# Rung 1: fixed rules per assay (these may use annotations; data_prep/build_annotation_tracks.py bits)

ANNOT = {"CGI": 1, "TSS_2KB": 2, "GENE_BODY": 4, "REPEAT": 8, "ZNF": 16, "CTCF_MOTIF": 32}
METH_HALF, MIN_CPG = 500, 2       # fiber CpG methylation over +- 500 bp, from >= 2 covered CpGs


def rule_scores(x, annot, assays):
    """Rung 1 rules (reports/TODO_extended.md T11), one 0 / 1 track per assay and fiber. x [B, 4, L, N] (m6A, CpG,
    MSP, nucleosome), annot [B, L] uint8 bits -> [B, K, L, N] float32. "Open region" = a large MSP (run >= 151 bp)."""
    B, C, L, N = x.shape
    with torch.autocast(device_type=x.device.type, enabled=False):
        f = x.float().permute(0, 3, 1, 2).reshape(B * N, C, L)
        m6a, cpg, msp, nuc = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
        bits = annot.long().unsqueeze(1).expand(B, N, L).reshape(B * N, L)
        has = {k: ((bits & v) > 0).float() for k, v in ANNOT.items()}
        big = large_runs(msp, BIG_MSP_HALF)
        cg_meth, cg_cov = (cpg > 0).float(), (cpg != 0).float() / 2
        n_cpg = boxsum(cg_cov, METH_HALF)
        meth = ratio(boxsum(cg_meth, METH_HALF), n_cpg)
        known = n_cpg >= MIN_CPG
        unmeth03, unmeth05, meth07 = (known & (meth < 0.3)).float(), (known & (meth < 0.5)).float(), \
            (known & (meth > 0.7)).float()
        # CTCF: a protected motif hit (motif m6A rate <= 0.5 x the 100 bp flanks, as the footprint test) inside or
        # touching an MSP, spread over the motif +- 100 bp
        m6_call, m6_info = (m6a > 0).float(), (m6a != 0).float()
        core_call, core_info = boxsum(m6_call, FP_CORE), boxsum(m6_info, FP_CORE)
        flank_rate = ratio(boxsum(m6_call, FP_FLANK) - core_call, boxsum(m6_info, FP_FLANK) - core_info)
        prot = ((core_info >= 4) & (flank_rate > 0) & (ratio(core_call, core_info) <= 0.5 * flank_rate)).float()
        ctcf = within(has["CTCF_MOTIF"] * prot * within(msp, 10), 100)
        rules = {
            "atac": big,
            "ctcf": ctcf,
            "h3k4me3": nuc * within(big, 300) * within(has["CGI"], 500) * unmeth03,
            "h3k27ac": nuc * within(big, 1000) * unmeth05,
            "h3k4me1": nuc * within(big, 2000) * unmeth05 * (1 - has["TSS_2KB"]),
            "h3k36me3": has["GENE_BODY"] * nuc * meth07,
            "h3k27me3": within(has["CGI"], 1000) * unmeth03 * (1 - within(big, 500)),
            "h3k9me3": ((has["REPEAT"] + has["ZNF"]) > 0).float() * meth07 * (1 - within(big, 1000)),
        }
        missing = [a for a in assays if a not in rules]
        if missing:
            raise ValueError(f"No rule for {missing}")
        out = torch.stack([rules[a] for a in assays], dim=1)                 # [BN, K, L]
        return out.reshape(B, N, len(assays), L).permute(0, 2, 3, 1).contiguous()


class FeatureMLP(nn.Module):
    """Rung 2: per-position MLP on FiberFeatures (no spatial filters beyond the fixed windows), the same Softplus
    output and coverage-normalized fiber averaging as the network. Built through models.MLPBaseline (BaseModel)."""

    def __init__(self, n_out, hidden=64, layers=2):
        super().__init__()
        dims = [N_FEATURES] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Conv1d(a, b, kernel_size=1), nn.GELU()]
        mods.append(nn.Conv1d(dims[-1], n_out, kernel_size=1))
        self.net = nn.Sequential(*mods)
        # No batch norm: padded fiber rows (all zeros) would skew its statistics; the features are in [0, 1] apart
        # from CpG observed / expected (<= 3), so the first layer handles the scales.

    def forward(self, feats):
        """feats [BN, N_FEATURES, L] -> [BN, K, L] raw outputs (before Softplus)."""
        return self.net(feats)
