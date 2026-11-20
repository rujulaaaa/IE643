**D-44 verdict** (n = 12 pairs, scenes a lighthouse on a rocky cliff, a path through a pine forest, a red sandstone desert canyon)

- HR1 FAIL: all-heads transfer 0.00 [-0.70, 0.74] (median 0.07), leak -0.01 [-0.09, 0.06]
- HR2 FAIL: attn_down_blocks.2 transfer -0.06 [-0.89, 0.78] (median 0.01); minus half of all heads -0.06 [-0.54, 0.42]
- HR3 FAIL: attn_h6 transfer -0.48 [-1.43, 0.41] (median -0.02); minus half of all heads -0.48 [-1.32, 0.14]

| arm | transfer | median | leak |
|---|---|---|---|
| attn_down_blocks.0 | -0.47 [-1.09, 0.11] | -0.10 | -0.10 [-0.19, -0.01] |
| attn_down_blocks.1 | 0.01 [-0.84, 0.99] | 0.00 | 0.00 [-0.08, 0.07] |
| attn_down_blocks.2 | -0.06 [-0.87, 0.75] | 0.01 | 0.05 [-0.02, 0.13] |
| attn_down_blocks.3 | 0.05 [-0.05, 0.17] | 0.00 | 0.02 [0.01, 0.03] |
| attn_first_third | 0.00 [-0.69, 0.74] | 0.07 | -0.01 [-0.10, 0.06] |
| attn_h0 | -0.45 [-1.01, 0.05] | -0.05 | -0.02 [-0.09, 0.04] |
| attn_h1 | -0.20 [-0.87, 0.45] | -0.01 | -0.00 [-0.04, 0.02] |
| attn_h2 | -0.08 [-0.50, 0.43] | -0.12 | -0.05 [-0.12, 0.01] |
| attn_h3 | -0.33 [-1.02, 0.38] | -0.08 | -0.02 [-0.07, 0.04] |
| attn_h4 | -0.40 [-0.99, 0.11] | -0.03 | -0.02 [-0.10, 0.05] |
| attn_h5 | -0.20 [-0.66, 0.22] | -0.03 | 0.05 [-0.04, 0.14] |
| attn_h6 | -0.48 [-1.43, 0.40] | -0.02 | -0.04 [-0.10, 0.01] |
| attn_h7 | -0.31 [-0.99, 0.36] | -0.01 | 0.02 [-0.00, 0.05] |
| attn_mid_block | -0.14 [-0.32, 0.02] | 0.00 | 0.00 [-0.01, 0.02] |
| attn_up_blocks.0 | 0.00 [-0.35, 0.46] | -0.08 | -0.02 [-0.04, 0.01] |
| attn_up_blocks.1 | -0.29 [-0.92, 0.35] | -0.08 | -0.01 [-0.04, 0.02] |
| attn_up_blocks.2 | 0.07 [-1.16, 1.35] | 0.00 | 0.02 [-0.02, 0.06] |
| attn_up_blocks.3 | -0.04 [-0.19, 0.09] | -0.01 | 0.01 [-0.02, 0.04] |

Per-scene mean all-heads transfer: a lighthouse on a rocky cliff -0.88, a path through a pine forest 0.83, a red sandstone desert canyon 0.06

**Knock-out (A<-B), original Part A pairs**

| candidate | knock-out | median | n |
|---|---|---|---|
| exp1_region | 0.17 [0.06, 0.34] | 0.11 | 12 |
| up1_motion_modules | 0.90 [0.68, 1.15] | 0.90 | 12 |
| up1_motion_modules_all | 0.91 [0.76, 1.07] | 0.94 | 12 |
| mm_down_blocks.0 | 0.88 [0.70, 1.03] | 0.95 | 12 |
| mm_down_blocks.1 | 0.86 [0.69, 1.01] | 0.96 | 12 |
| mm_down_blocks.2 | 0.81 [0.61, 1.01] | 0.87 | 12 |
| mm_down_blocks.3 | 0.55 [0.41, 0.71] | 0.47 | 12 |
| mm_mid_block | 0.54 [0.35, 0.75] | 0.42 | 12 |
| mm_up_blocks.0 | 0.73 [0.55, 0.89] | 0.80 | 12 |
| mm_up_blocks.1 | 0.91 [0.75, 1.07] | 0.94 | 12 |
| mm_up_blocks.2 | 0.90 [0.79, 1.02] | 0.97 | 12 |
| mm_up_blocks.3 | 1.03 [0.96, 1.11] | 1.00 | 12 |
