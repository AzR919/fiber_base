# 23rd Oct

First results on the new pipeline (setup in `Oct_19.md`): the dev-split model trained for 25 and 100 epochs,
compared with the cross-cell-average baseline, on all 8 cell types.

**Setup**
 - Training: the 5 dev cell types (GM12878, K562, HepG2, Panc1, H1; 200U samples), chr20, half cCRE-centred and
   half random regions; mid 8-assay UNet (`model15`), batch 8 × 1000 samples per epoch.
 - Short run: 25 epochs (`train02`). Long run: 100 epochs (`train03`); validation Pearson plateaus at ~0.49 by
   epoch ~60.
 - Test: all 6,843 valid 5 kb tiles of chr21 (never trained on), for the 5 dev cell types and the 3 held-out ones
   (HCT116, MCF-7, H9, never seen in training). H1 and H9 have no ATAC.
 - Scored at base-pair resolution on asinh(fold change), at positions covered by ≥ 5 fibers.

**Metrics** (full definitions in `metrics/README.md` and `metrics/APPENDIX.md`)
 - **Pearson r:** correlation between prediction and target, pooled over every scored base pair of chr21. Measures
   whether the predicted *shape* along the genome is right (insensitive to scale). Averaged over assays per cell.
 - **MSE:** mean squared error over the same positions. Also measures whether signal *levels* are right.
 - **Baseline:** for each cell, the average of the *other* dev cell types' tracks at the same positions. It needs no
   information about the cell at all, and it is strong because much of every track (e.g. housekeeping promoters)
   is the same in every cell type.
 - **Cell-type r** (cell-specific r): the partial correlation of prediction and target given the baseline. Both are
   first stripped of what the baseline explains (by regression), then correlated. It asks: *where this cell
   differs from the average cell, does the prediction differ the same way?* 0 = nothing beyond the average (the
   baseline itself scores exactly 0); its square ≈ the share of the cell-specific variation the model explains.
   Since the DNA input is the same for every cell type, a positive value means the model reads cell-specific
   information from the fibers.

### Summary: mean over assays, all scored chr21 bases

Long (100 epochs) / short (25 epochs) / baseline. **Bold** = better of the long run and the baseline.

| Cell type | Pearson r | MSE | Cell-type r (long / short) |
|---|---|---|---|
| GM12878 | 0.43 / 0.38 / **0.44** | **0.228** / 0.245 / 0.258 | +0.26 / +0.21 |
| K562 | 0.50 / 0.41 / **0.51** | **0.172** / 0.184 / 0.191 | +0.32 / +0.23 |
| HepG2 | **0.57** / 0.51 / 0.54 | **0.232** / 0.256 / 0.244 | +0.41 / +0.36 |
| Panc1 | 0.40 / 0.38 / **0.49** | **0.194** / 0.192 / 0.195 | +0.21 / +0.20 |
| H1 | **0.40** / 0.36 / 0.39 | **0.261** / 0.270 / 0.283 | +0.28 / +0.24 |
| HCT116 *(held out)* | 0.47 / 0.45 / **0.56** | 0.177 / 0.182 / **0.163** | +0.26 / +0.25 |
| MCF-7 *(held out)* | 0.45 / 0.41 / **0.51** | **0.162** / 0.166 / 0.166 | +0.27 / +0.23 |
| H9 *(held out)* | 0.38 / 0.36 / **0.47** | 0.177 / 0.176 / 0.182 | +0.19 / +0.18 |

 - Training 4× longer improves Pearson in every cell (+0.02 to +0.09) and the cell-type r in every cell.
 - **Pearson:** the long run beats the baseline in HepG2 and H1, is within 0.01 in GM12878 and K562, and still
   trails in Panc1 and the three held-out cell types.
 - **MSE:** the long run beats the baseline in 6 of 8 cell types (matching it in Panc1), trailing only in HCT116
   and H9. Pearson and MSE disagree because the average gets the *shape* along the genome right (shared
   positions) while the model gets each cell's *levels* closer.
 - **Cell-type r** is positive in every cell type, +0.19 to +0.41, and as high in the held-out cell types
   (+0.19 to +0.27) as in several dev ones: the model captures cell-specific signal that generalizes to new cell
   types, which the plain Pearson comparison hides.

### Pearson r per assay (long / short / baseline)

| Cell type | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| GM12878 | .73/.70/.68 | .43/.40/.45 | .56/.51/.35 | .30/.25/.38 | .55/.50/.41 | .39/.18/.51 | .22/.20/.36 | .26/.26/.36 |
| K562 | .72/.69/.76 | .57/.53/.50 | .48/.43/.43 | .59/.47/.54 | .36/.33/.40 | .63/.26/.65 | .32/.29/.42 | .31/.31/.39 |
| HepG2 | .82/.80/.78 | .41/.40/.42 | .50/.47/.45 | .57/.42/.55 | .67/.63/.50 | .47/.26/.63 | .63/.59/.48 | .51/.50/.48 |
| Panc1 | .64/.63/.72 | .58/.52/.56 | .47/.45/.45 | .26/.20/.34 | .37/.36/.46 | .11/.13/.40 | .35/.36/.42 | .44/.42/.54 |
| H1 | – | .67/.64/.58 | .35/.31/.30 | .30/.21/.29 | .47/.45/.36 | .24/.21/.43 | .31/.27/.37 | .50/.46/.43 |
| HCT116 | .74/.72/.73 | .60/.56/.59 | .55/.52/.49 | .26/.25/.48 | .54/.51/.54 | .18/.18/.55 | .58/.55/.66 | .30/.32/.48 |
| MCF-7 | .72/.71/.65 | .64/.60/.60 | .38/.36/.37 | .26/.21/.33 | .56/.53/.56 | .34/.22/.63 | .31/.30/.49 | .36/.34/.41 |
| H9 | – | .59/.56/.59 | .35/.30/.36 | .38/.34/.36 | .47/.45/.50 | .16/.16/.45 | .29/.27/.45 | .40/.44/.57 |

### MSE per assay (long / short / baseline)

| Cell type | ATAC | H3K4me3 | H3K27ac | H3K27me3 | H3K4me1 | H3K36me3 | H3K9me3 | CTCF |
|---|---|---|---|---|---|---|---|---|
| GM12878 | .066/.079/.091 | .160/.182/.227 | .259/.295/.374 | .283/.290/.266 | .186/.198/.249 | .329/.381/.292 | .310/.308/.333 | .228/.229/.231 |
| K562 | .084/.088/.094 | .190/.188/.226 | .140/.131/.171 | .163/.192/.198 | .165/.154/.208 | .187/.272/.162 | .230/.240/.244 | .213/.209/.224 |
| HepG2 | .061/.077/.066 | .220/.228/.254 | .170/.174/.191 | .139/.169/.147 | .232/.268/.300 | .171/.203/.145 | .676/.730/.650 | .186/.201/.196 |
| Panc1 | .117/.125/.083 | .210/.218/.226 | .222/.214/.217 | .197/.216/.227 | .247/.235/.220 | .297/.260/.261 | .155/.153/.221 | .103/.113/.103 |
| H1 | – | .115/.120/.139 | .266/.270/.316 | .353/.381/.377 | .304/.311/.341 | .341/.334/.306 | .273/.285/.299 | .176/.187/.205 |
| HCT116 | .079/.089/.089 | .125/.139/.139 | .161/.171/.184 | .191/.187/.169 | .146/.145/.148 | .227/.214/.181 | .304/.337/.229 | .184/.174/.164 |
| MCF-7 | .110/.122/.126 | .184/.185/.173 | .178/.176/.196 | .168/.167/.196 | .163/.168/.170 | .185/.195/.136 | .124/.126/.146 | .186/.194/.186 |
| H9 | – | .138/.142/.144 | .186/.181/.189 | .232/.249/.253 | .151/.143/.166 | .192/.178/.197 | .201/.210/.198 | .140/.133/.129 |

 - **Strongest:** ATAC (r 0.64–0.82), and the active marks H3K4me3, H3K27ac, H3K4me1, which beat the baseline in
   most cell types on both metrics. These are the marks most directly reflected in a molecule's accessibility.
 - **Weakest: H3K36me3.** Longer training fixed it in GM12878, K562 and HepG2 (e.g. K562 0.26 → 0.63), but in
   Panc1, HCT116 and H9 it stays at 0.11–0.18 against a 0.40–0.55 baseline. H3K36me3 covers whole gene bodies,
   far wider than a 5 kb window, so longer DNA context is the likely fix. H3K27me3 and H3K9me3 also mostly trail
   the baseline.
 - HepG2's H3K9me3 has much higher values than other cells, so every method has a large MSE there.

**Next:** the single-molecule tests, starting with mixture-only supervision (train on mixtures of two cell types,
recover each one from its own fibers), and the comparison ladder (plan in `TODO.md`).
