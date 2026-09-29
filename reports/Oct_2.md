# 2nd Oct

 - Values are per-assay average MSE loss on cCRES in Chr 21 (~11,400). Each model was trained on Chr 20 (~25,000 cCREs).
 - "Avg" is the composite loss across all 4 assays for that cell type.
 - **Bold** = lowest (best) loss in each column, *italic* = second-lowest, <u>underline</u> = third-lowest (rankings computed across all 7 rows).
 - Rows 4-6 are leave-one-out pairs: each is trained on two cell types and evaluated on all three, including the one withheld from training.

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
      <td>GM12878</td>
      <td style="border-left: 1px solid #ccc"><b>0.5324</b></td>
      <td><b>0.2987</b></td>
      <td><b>0.3126</b></td>
      <td><i>0.1822</i></td>
      <td><b>0.3315</b></td>
      <td style="border-left: 1px solid #ccc"><u>0.7677</u></td>
      <td>0.4513</td>
      <td><i>0.4657</i></td>
      <td>1.1487</td>
      <td>0.7084</td>
      <td style="border-left: 1px solid #ccc"><i>0.6141</i></td>
      <td>0.5814</td>
      <td>0.5086</td>
      <td>0.1198</td>
      <td>0.4560</td>
    </tr>
    <tr>
      <td>K562_200U</td>
      <td style="border-left: 1px solid #ccc">0.5767</td>
      <td>0.3583</td>
      <td>0.3739</td>
      <td>0.3793</td>
      <td>0.4220</td>
      <td style="border-left: 1px solid #ccc">0.7826</td>
      <td><i>0.3275</i></td>
      <td>0.4848</td>
      <td><b>0.6483</b></td>
      <td><b>0.5608</b></td>
      <td style="border-left: 1px solid #ccc">0.6421</td>
      <td><u>0.3973</u></td>
      <td>0.4852</td>
      <td>0.1802</td>
      <td>0.4262</td>
    </tr>
    <tr>
      <td>HepG2_200U</td>
      <td style="border-left: 1px solid #ccc">0.8180</td>
      <td>0.4713</td>
      <td>0.3609</td>
      <td>0.1871</td>
      <td>0.4593</td>
      <td style="border-left: 1px solid #ccc">0.8230</td>
      <td><b>0.3259</b></td>
      <td>0.4850</td>
      <td>0.9142</td>
      <td>0.6370</td>
      <td style="border-left: 1px solid #ccc">0.7658</td>
      <td><b>0.3564</b></td>
      <td><b>0.4515</b></td>
      <td><b>0.1126</b></td>
      <td>0.4216</td>
    </tr>
    <tr>
      <td>GM12878+K562_200U <br><small>(holds out HepG2_200U)</small></td>
      <td style="border-left: 1px solid #ccc">0.5605</td>
      <td><i>0.3202</i></td>
      <td>0.3458</td>
      <td>0.1976</td>
      <td><u>0.3560</u></td>
      <td style="border-left: 1px solid #ccc">0.7875</td>
      <td>0.3630</td>
      <td>0.4744</td>
      <td>0.8908</td>
      <td>0.6289</td>
      <td style="border-left: 1px solid #ccc"><u>0.6184</u></td>
      <td>0.4333</td>
      <td><i>0.4724</i></td>
      <td>0.1274</td>
      <td><u>0.4129</u></td>
    </tr>
    <tr>
      <td>GM12878+HepG2_200U <br><small>(holds out K562_200U)</small></td>
      <td style="border-left: 1px solid #ccc">0.5609</td>
      <td>0.3533</td>
      <td><u>0.3353</u></td>
      <td><u>0.1856</u></td>
      <td>0.3588</td>
      <td style="border-left: 1px solid #ccc"><b>0.6869</b></td>
      <td><u>0.3484</u></td>
      <td><u>0.4658</u></td>
      <td>1.0397</td>
      <td>0.6352</td>
      <td style="border-left: 1px solid #ccc">0.6341</td>
      <td>0.4432</td>
      <td>0.4827</td>
      <td><u>0.1180</u></td>
      <td>0.4195</td>
    </tr>
    <tr>
      <td>K562_200U+HepG2_200U <br><small>(holds out GM12878)</small></td>
      <td style="border-left: 1px solid #ccc"><u>0.5555</u></td>
      <td><u>0.3245</u></td>
      <td>0.3484</td>
      <td>0.2044</td>
      <td>0.3582</td>
      <td style="border-left: 1px solid #ccc">0.7813</td>
      <td>0.3684</td>
      <td>0.4772</td>
      <td><u>0.8827</u></td>
      <td><u>0.6274</u></td>
      <td style="border-left: 1px solid #ccc"><b>0.6100</b></td>
      <td>0.4351</td>
      <td><u>0.4733</u></td>
      <td>0.1298</td>
      <td><b>0.4120</b></td>
    </tr>
    <tr>
      <td>GM12878+K562_200U+HepG2_200U</td>
      <td style="border-left: 1px solid #ccc"><i>0.5374</i></td>
      <td>0.3305</td>
      <td><i>0.3331</i></td>
      <td><b>0.1748</b></td>
      <td><i>0.3440</i></td>
      <td style="border-left: 1px solid #ccc"><i>0.7043</i></td>
      <td>0.3531</td>
      <td><b>0.4488</b></td>
      <td><i>0.8417</i></td>
      <td><i>0.5870</i></td>
      <td style="border-left: 1px solid #ccc">0.6523</td>
      <td><i>0.3896</i></td>
      <td>0.4900</td>
      <td><i>0.1170</i></td>
      <td><i>0.4122</i></td>
    </tr>
  </tbody>
</table>
