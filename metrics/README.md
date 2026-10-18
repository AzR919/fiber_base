# metrics/

Every number the pipeline reports, what it measures, and where to find it. Definitions, formulas, worked
examples and caveats are in [APPENDIX.md](APPENDIX.md). The code is in `metrics.py` (re-exported, so
`from metrics import ...`); `evaluator.py` only decides what to score.

All metrics are computed on **asinh(fold change)** signal (the space the model is trained in), per assay, at
**base-pair resolution** (no binning), and only at **scored positions**.

## Scored positions

A (sample, assay, position) is scored when the cell type has ENCODE data for the assay and the position is
covered by at least `min_coverage` fibers (default 5), counting each fiber only inside its scored span (its
alignment trimmed to the first..last nucleosome). Everything below, training loss included, uses exactly these
positions. → Appendix A1

## Metrics

| Metric | Question it answers | Where it appears | Appendix |
|---|---|---|---|
| **Training / validation loss** | How far is the bulk prediction from the target? (what the model optimises) | train/val loss per epoch (console, wandb) | A2 |
| **MSE** | How close are the predicted signal *levels*, per assay? | eval: `per_assay_loss`, `loss…` | A3 |
| **Pearson r** | Does the predicted *shape* along the genome follow the target? Pooled over every scored base pair | eval: `per_assay_pearson`, `pearson…`; val: `val_pearson` | A4 |
| **By region type** | Where does the model do well: promoters, enhancers, CTCF sites, outside cCREs? | eval `strata` (console tables, JSON, `pearson@<type>`) | A5 |
| **Per-window loss and r** | How does one window look? (for reading dashboards, not for comparing models) | dashboard panel titles | A6 |
| **Cross-cell-average baseline** | What does "the average of the other training cells" score? The reference a model must beat | `eval.py --baseline(_only)`: `baseline` block, `baseline_pearson…` | A7 |
| **Cell-specific r** (partial correlation) | Does the model explain this cell's *own* signal beyond the cross-cell average? | `eval.py --baseline`: `partial` block, `partial_pearson…` | A8 |
| *Pairwise differential r, fiber-origin AUC* | *Do per-molecule outputs separate two cells in the right direction?* | *planned (test 1)* | A9 |

## How to read them together

- **MSE and Pearson can disagree.** MSE rewards getting signal levels right; Pearson only shape. A smooth,
  averaged prediction can win on r and lose on MSE (we saw this: the dev-split model beats the 5-cell baseline
  on MSE in most cells but not on r).
- **Plain r mixes shared and cell-specific signal.** Much of each track (housekeeping promoters, shared
  accessible sites) is the same in every cell, and the baseline gets that for free. The cell-specific r isolates
  what needs information about *this* cell, which can only come from the fibers (the reference DNA is the same
  for every cell).
- **None of these is a per-molecule test.** They all score the bulk (the average over molecules). Whether
  individual molecules are imputed correctly needs group-level tests (test 1 and later).
- Compare models only on the **same window set, cells and `min_coverage`** (the JSON records the window set
  and sample counts).
