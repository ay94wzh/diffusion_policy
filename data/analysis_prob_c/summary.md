# latent series — /data/zihan/runs/run_square_prob_c_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 1.51e-04 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.00903, zg 0.069, zg_n1 0.0717 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 1.1e-09, zg 8.9e-09, zg_n1 1.5e-08 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.28 | read against |
| chance level zg (unrelated) | rel 1.41, cos 0.99, cka 0.23, proc 1.02 | read against |
| chance level zg_n1 (unrelated) | rel 1.41, cos 0.99, cka 0.23, proc 1.00 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 30.26 | 16.61 | 14.32 | 13.99 | 14.03 | 79 | 14.03 ± 0.003896 |
| zv_rel_spread_mean | 0.02355 | 0.1155 | 0.136 | 0.137 | 0.134 | 73 | 0.1344 ± 0.0002718 |
| zv_rel_spread_min | 0.01728 | 0.1092 | 0.1282 | 0.1315 | 0.1304 | 80 | 0.1305 ± 5.192e-05 |
| zg_norm_mean | 30.38 | 15.28 | 13.25 | 12.96 | 13.03 | 73 | 13.03 ± 0.006732 |
| zg_rel_spread | 0.02229 | 0.1223 | 0.142 | 0.145 | 0.1436 | 78 | 0.1437 ± 0.0001098 |
| zg_n1_norm_mean | 30.63 | 15.33 | 13.26 | 13.04 | 13.18 | 70 | 13.16 ± 0.01088 |
| zg_n1_rel_spread | 0.02434 | 0.1218 | 0.1421 | 0.1435 | 0.143 | 78 | 0.1431 ± 7.977e-05 |
| zv_pair_ratio | 0.5782 | 0.2489 | 0.2591 | 0.2588 | 0.256 | 11 | 0.2562 ± 0.0001275 |
| zv_pr | 3.131 | 7.162 | 8.085 | 8.43 | 8.462 | 111 | 8.461 ± 0.0005053 |
| zg_pr | 1.963 | 6.927 | 7.849 | 8.165 | 8.172 | 102 | 8.175 ± 0.001772 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 65 | - | 4 | 1 | 11 |
| zg | 65 | - | 4 | 1 | 7 |
| zg_n1 | 67 | - | 4 | 1 | 7 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 81 | 138 | 145 | 153 | 152 |
| zg settle | 80 | 133 | 145 | 152 | 151 |
| zg_n1 settle | 78 | 136 | 145 | 153 | 153 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 14.04 | 0.1365 | 7.661 | 0.3646 | 1.055 | 0.3457 | 71 |
| 2 | 14.01 | 0.1369 | 7.757 | 0.3021 | 1.076 | 0.2807 | 67 |
| 3 | 13.98 | 0.1388 | 7.835 | 0.2673 | 1.09 | 0.2453 | 67 |
| 4 | 13.98 | 0.1382 | 7.856 | 0.2497 | 1.097 | 0.2275 | 66 |
| 5 | 14.03 | 0.1337 | 7.894 | 0.2434 | 1.102 | 0.221 | 65 |
| 6 | 14.04 | 0.1312 | 7.877 | 0.2486 | 1.102 | 0.2256 | 65 |
| 7 | 14.08 | 0.1333 | 7.92 | 0.2519 | 1.104 | 0.2282 | 65 |
| 8 | 14.09 | 0.1335 | 7.874 | 0.2597 | 1.102 | 0.2355 | 67 |
| 9 | 14.08 | 0.1309 | 7.854 | 0.2709 | 1.101 | 0.246 | 66 |
| 10 | 14.08 | 0.1306 | 7.865 | 0.2913 | 1.098 | 0.2652 | 65 |
| 11 | 13.95 | 0.1304 | 7.869 | 0.3272 | 1.091 | 0.3 | 65 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.08096 | 0.9629 | 0.01384 | 0.9992 | 0.9582 | 0.9992 |
| 50 | 0.1999 | 0.9975 | 0.2175 | 0.9984 | 0.9959 | 0.9969 |
| 100 | 0.2055 | 0.9978 | 0.2529 | 0.9982 | 0.9965 | 0.9972 |
| 150 | 0.1935 | 0.9979 | 0.2727 | 0.9974 | 0.9962 | 0.9969 |
| 200 | 0.1777 | 0.9976 | 0.2895 | 0.9959 | 0.9952 | 0.996 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 176 | 0.08096 | 0.1777 | 0.9976 |
| n1_vs_zv_az0 | 157 | 0.01384 | 0.2895 | 0.9959 |
| full_vs_zv_az0 | 105 | 0.0821 | 0.3442 | 0.9952 |
| full_vs_mean_pool | 115 | 0.01431 | 0.3361 | 0.996 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.620 | 50 |
| 100 | 0.780 | 50 |
| 150 | 0.800 | 50 |
| 200 | 0.840 | 50 |

behaviour settles at epoch not_settled (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
