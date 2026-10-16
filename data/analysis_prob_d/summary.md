# latent series — /data/zihan/runs/run_square_prob_d_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 1.64e-04 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.00876, zg 0.0665, zg_n1 0.0703 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 3.1e-09, zg 1.1e-08, zg_n1 2.6e-08 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.28 | read against |
| chance level zg (unrelated) | rel 1.41, cos 0.99, cka 0.22, proc 1.03 | read against |
| chance level zg_n1 (unrelated) | rel 1.41, cos 0.99, cka 0.23, proc 1.00 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 30.24 | 15.28 | 12.58 | 12.16 | 12.18 | 82 | 12.18 ± 0.004885 |
| zv_rel_spread_mean | 0.02367 | 0.1183 | 0.1478 | 0.1558 | 0.1581 | 112 | 0.1581 ± 6.209e-05 |
| zv_rel_spread_min | 0.01717 | 0.1097 | 0.1352 | 0.1425 | 0.1445 | 111 | 0.1444 ± 6.678e-05 |
| zg_norm_mean | 30.35 | 13.65 | 11.35 | 11.05 | 11.15 | 73 | 11.14 ± 0.0105 |
| zg_rel_spread | 0.02246 | 0.1317 | 0.1628 | 0.1704 | 0.1718 | 105 | 0.1719 ± 2.537e-05 |
| zg_n1_norm_mean | 30.6 | 13.95 | 11.67 | 11.38 | 11.48 | 72 | 11.47 ± 0.01123 |
| zg_n1_rel_spread | 0.02419 | 0.1221 | 0.155 | 0.1632 | 0.1649 | 109 | 0.1649 ± 2.673e-05 |
| zv_pair_ratio | 0.5813 | 0.2249 | 0.2385 | 0.2367 | 0.2347 | 32 | 0.2348 ± 8.122e-05 |
| zv_pr | 3.144 | 6.828 | 7.699 | 7.993 | 8.025 | 110 | 8.023 ± 0.002117 |
| zg_pr | 1.961 | 6.5 | 7.384 | 7.77 | 7.827 | 116 | 7.826 ± 0.001241 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 67 | - | 4 | 1 | 11 |
| zg | 62 | - | 4 | 1 | 7 |
| zg_n1 | 71 | - | 3 | 1 | 7 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 84 | 138 | 146 | 152 | 152 |
| zg settle | 78 | 133 | 146 | 152 | 151 |
| zg_n1 settle | 77 | 135 | 147 | 154 | 154 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 12.06 | 0.174 | 7.212 | 0.3207 | 1.055 | 0.3039 | 68 |
| 2 | 12.12 | 0.172 | 7.366 | 0.2679 | 1.073 | 0.2498 | 67 |
| 3 | 12.12 | 0.1684 | 7.466 | 0.2401 | 1.083 | 0.2217 | 68 |
| 4 | 12.16 | 0.1629 | 7.6 | 0.2278 | 1.09 | 0.209 | 68 |
| 5 | 12.2 | 0.1585 | 7.693 | 0.2259 | 1.094 | 0.2066 | 65 |
| 6 | 12.23 | 0.155 | 7.679 | 0.2322 | 1.095 | 0.212 | 65 |
| 7 | 12.24 | 0.1504 | 7.719 | 0.2363 | 1.095 | 0.2158 | 65 |
| 8 | 12.27 | 0.152 | 7.742 | 0.2412 | 1.095 | 0.2202 | 67 |
| 9 | 12.24 | 0.1535 | 7.689 | 0.249 | 1.096 | 0.2273 | 68 |
| 10 | 12.23 | 0.1483 | 7.762 | 0.266 | 1.093 | 0.2433 | 67 |
| 11 | 12.13 | 0.1445 | 7.687 | 0.2986 | 1.086 | 0.2749 | 67 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.0821 | 0.962 | 0.01364 | 0.9992 | 0.9574 | 0.9992 |
| 50 | 0.1446 | 0.9973 | 0.2407 | 0.9986 | 0.9962 | 0.9985 |
| 100 | 0.1613 | 0.9968 | 0.267 | 0.9979 | 0.9954 | 0.9977 |
| 150 | 0.1611 | 0.997 | 0.2835 | 0.9966 | 0.9954 | 0.9967 |
| 200 | 0.1585 | 0.9968 | 0.301 | 0.9942 | 0.9938 | 0.9947 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 127 | 0.0821 | 0.1585 | 0.9968 |
| n1_vs_zv_az0 | 157 | 0.01364 | 0.301 | 0.9942 |
| full_vs_zv_az0 | 150 | 0.08326 | 0.3293 | 0.9938 |
| full_vs_mean_pool | 154 | 0.01416 | 0.3067 | 0.9947 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.660 | 50 |
| 100 | 0.820 | 50 |
| 150 | 0.800 | 50 |
| 200 | 0.820 | 50 |

behaviour settles at epoch 100 (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
