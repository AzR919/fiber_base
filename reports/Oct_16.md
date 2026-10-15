# 16th Oct

 - Values are per-assay average MSE loss on cCRES in Chr 21 (~11,400). Each model was trained on Chr 20 (~25,000 cCREs) on all three cell types (GM12878, K562_200U, HepG2_200U).
 - "Avg" is the composite loss across all 4 assays for that cell type.
 - Both models are identical (same data, hyperparameters, seed) except for the input: *with fire_msp* uses all five input channels (m6a, cpg, msp, nuc, fire_msp), *without fire_msp* drops the fire_msp channel (four inputs).
 - **Bold** = lowest (best) loss in each column, *italic* = second-lowest (rankings computed across the 2 models).
 - Δ = (without fire_msp) − (with fire_msp), computed from unrounded values. Negative = lower (better) loss without fire_msp.

<table>
  <thead>
    <tr>
      <th rowspan="3", style="text-align: center, vertical-align: bottom">Model Trained on</th>
      <th colspan="15", style="text-align: center">Cell types</th>
    </tr>
    <tr>
      <th colspan="5", style="text-align: center">GM12878</th>
      <th colspan="5", style="text-align: center">K562_200U</th>
      <th colspan="5", style="text-align: center">HepG2_200U</th>
    </tr>
    <tr>
      <th style="border-left: 1px solid #ccc">ATAC</th>
      <th>H3K4me3</th>
      <th>H3K27ac</th>
      <th>H3K27me3</th>
      <th>Avg</th>
      <th style="border-left: 1px solid #ccc">ATAC</th>
      <th>H3K4me3</th>
      <th>H3K27ac</th>
      <th>H3K27me3</th>
      <th>Avg</th>
      <th style="border-left: 1px solid #ccc">ATAC</th>
      <th>H3K4me3</th>
      <th>H3K27ac</th>
      <th>H3K27me3</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GM12878+K562_200U+HepG2_200U <br><small>(with fire_msp)</small></td>
      <td style="border-left: 1px solid #ccc"><i>0.5374</i></td>
      <td><i>0.3305</i></td>
      <td><i>0.3331</i></td>
      <td><b>0.1748</b></td>
      <td><i>0.3440</i></td>
      <td style="border-left: 1px solid #ccc"><i>0.7043</i></td>
      <td><i>0.3531</i></td>
      <td><i>0.4488</i></td>
      <td><b>0.8417</b></td>
      <td><i>0.5870</i></td>
      <td style="border-left: 1px solid #ccc"><b>0.6523</b></td>
      <td><b>0.3896</b></td>
      <td><b>0.4900</b></td>
      <td><b>0.1170</b></td>
      <td><b>0.4122</b></td>
    </tr>
    <tr>
      <td>GM12878+K562_200U+HepG2_200U <br><small>(without fire_msp)</small></td>
      <td style="border-left: 1px solid #ccc"><b>0.5084</b></td>
      <td><b>0.3092</b></td>
      <td><b>0.3179</b></td>
      <td><i>0.1858</i></td>
      <td><b>0.3303</b></td>
      <td style="border-left: 1px solid #ccc"><b>0.6945</b></td>
      <td><b>0.3090</b></td>
      <td><b>0.4090</b></td>
      <td><i>0.8686</i></td>
      <td><b>0.5703</b></td>
      <td style="border-left: 1px solid #ccc"><i>0.6601</i></td>
      <td><i>0.4184</i></td>
      <td><i>0.4990</i></td>
      <td><i>0.1226</i></td>
      <td><i>0.4250</i></td>
    </tr>
    <tr>
      <td>Δ <small>(without − with)</small></td>
      <td style="border-left: 1px solid #ccc; border-top: 1px solid #ccc">−0.0290</td>
      <td style="border-top: 1px solid #ccc">−0.0213</td>
      <td style="border-top: 1px solid #ccc">−0.0152</td>
      <td style="border-top: 1px solid #ccc">+0.0110</td>
      <td style="border-top: 1px solid #ccc">−0.0136</td>
      <td style="border-left: 1px solid #ccc; border-top: 1px solid #ccc">−0.0098</td>
      <td style="border-top: 1px solid #ccc">−0.0441</td>
      <td style="border-top: 1px solid #ccc">−0.0399</td>
      <td style="border-top: 1px solid #ccc">+0.0269</td>
      <td style="border-top: 1px solid #ccc">−0.0167</td>
      <td style="border-left: 1px solid #ccc; border-top: 1px solid #ccc">+0.0077</td>
      <td style="border-top: 1px solid #ccc">+0.0288</td>
      <td style="border-top: 1px solid #ccc">+0.0090</td>
      <td style="border-top: 1px solid #ccc">+0.0056</td>
      <td style="border-top: 1px solid #ccc">+0.0128</td>
    </tr>
  </tbody>
</table>
