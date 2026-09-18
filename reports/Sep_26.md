# 18 Sep

## Progress
  - Bug Fixes
    - Not marking locations beyond the fiber boundaries
    - Instead of restricting to averaging using number of fibers, now have a coverage tensor of shape L containing the depth at each base pair
  - Additions
    - Arbitrary number of output assays
      - Chosen 4 for now: "atac", "h3k4me3", "h3k27ac", "h3k27me3"
    - Train and eval arbitrary number of cell types
      - Chosen 3 for now: GM12878, K562_200U, HepG2_200U

## TODO

Fill up this table

<table>
  <thead>
    <tr>
      <th rowspan="3", style="text-align: center, vertical-align: bottom">Model Trained on</th>
      <th colspan="9", style="text-align: center">Cell types</th>
    </tr>
    <tr>
      <th colspan="3", style="text-align: center">GM12878</th>
      <th colspan="3", style="text-align: center">K562</th>
      <th colspan="3", style="text-align: center">HepG2_200U</th>
    </tr>
    <tr>
      <th style="border-left: 1px solid #ccc">H3K4me3</th>
      <th>H3K27ac</th>
      <th>H3K27me3</th>
      <th style="border-left: 1px solid #ccc">H3K4me3</th>
      <th>H3K27ac</th>
      <th>H3K27me3</th>
      <th style="border-left: 1px solid #ccc">H3K4me3</th>
      <th>H3K27ac</th>
      <th style="border-right: 1px solid #ccc">H3K27me3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GM12878</td>
    </tr>
    <tr>
      <td>K562</td>
    </tr>
    <tr>
      <td>GM12878+K562 *</td>
    </tr>
    <tr>
      <td>GM12878+K562+HepG2_200U</td>
    </tr>
  </tbody>
</table>
