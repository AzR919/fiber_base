# 25th Sep

 - Values are per-assay average MSE loss on cCRES in Chr 21 (~11,400). Each model was trained on Chr 20 (~25,000 cCREs).
 - "Avg" is the composite loss across all 4 assays for that cell type.
 - **Bold** = lowest (best) loss in each column, *italic* = second-lowest, <u>underline</u> = third-lowest.

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
      <td style="border-left: 1px solid #ccc"><i>0.7677</i></td>
      <td>0.4513</td>
      <td><i>0.4657</i></td>
      <td>1.1487</td>
      <td>0.7084</td>
      <td style="border-left: 1px solid #ccc"><b>0.6141</b></td>
      <td>0.5814</td>
      <td>0.5086</td>
      <td><u>0.1198</u></td>
      <td>0.4560</td>
    </tr>
    <tr>
      <td>K562_200U</td>
      <td style="border-left: 1px solid #ccc">0.5767</td>
      <td>0.3583</td>
      <td>0.3739</td>
      <td>0.3793</td>
      <td>0.4220</td>
      <td style="border-left: 1px solid #ccc"><u>0.7826</u></td>
      <td><i>0.3275</i></td>
      <td>0.4848</td>
      <td><b>0.6483</b></td>
      <td><b>0.5608</b></td>
      <td style="border-left: 1px solid #ccc"><u>0.6421</u></td>
      <td><u>0.3973</u></td>
      <td><u>0.4852</u></td>
      <td>0.1802</td>
      <td>0.4262</td>
    </tr>
    <tr>
      <td>HepG2_200U</td>
      <td style="border-left: 1px solid #ccc">0.8180</td>
      <td>0.4713</td>
      <td>0.3609</td>
      <td><u>0.1871</u></td>
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
      <td><u>0.4216</u></td>
    </tr>
    <tr>
      <td>GM12878+K562_200U *</td>
      <td style="border-left: 1px solid #ccc"><u>0.5605</u></td>
      <td><i>0.3202</i></td>
      <td><u>0.3458</u></td>
      <td>0.1976</td>
      <td><u>0.3560</u></td>
      <td style="border-left: 1px solid #ccc">0.7875</td>
      <td>0.3630</td>
      <td><u>0.4744</u></td>
      <td><u>0.8908</u></td>
      <td><u>0.6289</u></td>
      <td style="border-left: 1px solid #ccc"><i>0.6184</i></td>
      <td>0.4333</td>
      <td><i>0.4724</i></td>
      <td>0.1274</td>
      <td><i>0.4129</i></td>
    </tr>
    <tr>
      <td>GM12878+K562_200U+HepG2_200U</td>
      <td style="border-left: 1px solid #ccc"><i>0.5374</i></td>
      <td><u>0.3305</u></td>
      <td><i>0.3331</i></td>
      <td><b>0.1748</b></td>
      <td><i>0.3440</i></td>
      <td style="border-left: 1px solid #ccc"><b>0.7043</b></td>
      <td><u>0.3531</u></td>
      <td><b>0.4488</b></td>
      <td><i>0.8417</i></td>
      <td><i>0.5870</i></td>
      <td style="border-left: 1px solid #ccc">0.6523</td>
      <td><i>0.3896</i></td>
      <td>0.4900</td>
      <td><i>0.1170</i></td>
      <td><b>0.4122</b></td>
    </tr>
  </tbody>
</table>


