# 19th Oct

Infrastructure week: no new results yet; the next runs use everything below.

**Data**
 - All bulk targets are now ENCODE *fold change over control* bigWigs (GRCh38), selected and downloaded automatically by one command (`data_prep/setup_data.py`), reproducible via a lock file.
 - 8 assays: ATAC, H3K4me3, H3K27ac, H3K27me3, H3K4me1, H3K36me3, H3K9me3, CTCF.
 - 8 cell lines: GM12878, K562, HepG2, HCT116, MCF-7, Panc1 (all 8 assays) and H1, H9 (no ATAC). Assays a cell line lacks are masked: no loss, no gradient.
 - Only the 200U Fiber-seq samples are used for now.

**Cell-type split**
 - Dev (ablations, model selection): GM12878, K562, HepG2, Panc1, H1.
 - Held-out (out-of-distribution test only): HCT116, MCF-7, H9.
 - Leave-one-out data configs for the dev set (`data21`–`data25`), all 5 dev cells for final models (`data20`).

**Train / val / test regions**
 - Train: chr20, half cCRE-centred regions, half uniform random regions (previously cCREs only).
 - Val: fixed set of 200 chr22 regions (previously 1,000 chr21 cCREs per cell, which was ~75% of run time).
 - Test: all 6,843 non-overlapping 5 kb tiles of chr21 (previously ~11.4k overlapping windows centred on cCREs), plus a 2,000-tile mini set. Blacklisted regions and assembly gaps are excluded.
 - Metrics are also reported per region type: bases in promoter / enhancer / CTCF cCREs and outside cCREs.
 - `data26` (GM12878 + K562, cCRE-only training, tested on HepG2) is a temporary run for a rough comparison with the old setup.

**Metrics and plots**
 - Pearson r (pooled over every base pair, no binning) added next to MSE, for validation each epoch and for final eval.
 - Dashboards redrawn for 8 assays with larger fonts; each assay panel shows the window's loss and r.

**Models**
 - 8-assay UNet03 configs in three sizes, each with a no-fire_msp twin: small (`model13`/`14`, 1g.10gb), mid (`model15`/`16`, 2g.20gb), large (`model17`/`18`, 2g.20gb).

**Data loading**
 - **Bug fix:** m6a and CpG call positions were stored as float32, which cannot represent coordinates above 16.7 Mb exactly. Calls were shifted 1–2 bp (about half of them on chr21) and ~15% were lost where two calls rounded to the same base. Inputs are now exact, so new results are not directly comparable with earlier runs.
 - Loading a sample takes ~155 ms instead of ~270 ms (redundant per-fiber work removed), and samples now load in parallel DataLoader workers.

**Training objective**
 - Each fiber now only counts inside its own span, trimmed to its first..last nucleosome (read ends beyond them have no MSP/nucleosome calls). Before, every fiber row produced output over the whole window, including where the fiber does not exist and in padding rows; those outputs were summed into the bulk and received gradient, so the model could put signal on empty rows (e.g. from the reference DNA alone). Now they are zeroed and get no gradient. Trimming removes ~2% of aligned bases.
 - Loss and metrics only use positions covered by at least 5 fibers.
