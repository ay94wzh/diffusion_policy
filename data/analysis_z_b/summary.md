# latent series — /data/zihan/runs/run_square_zb_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 1.96e-04 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.0072, zg 0.0169, zg_n1 0.0167 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 4.2e-09, zg 4e-08, zg_n1 5.7e-08 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.30 | read against |
| chance level zg (unrelated) | rel 1.41, cos 1.00, cka 0.09, proc 1.80 | read against |
| chance level zg_n1 (unrelated) | rel 1.41, cos 1.00, cka 0.09, proc 1.80 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 34.45 | 11.91 | 9.332 | 8.832 | 8.852 | 73 | 8.845 ± 0.004832 |
| zv_rel_spread_mean | 0.02636 | 0.2052 | 0.2475 | 0.2464 | 0.2366 | 144 | 0.2376 ± 0.0007806 |
| zv_rel_spread_min | 0.02155 | 0.1965 | 0.2367 | 0.2381 | 0.2278 | 150 | 0.2292 ± 0.001015 |
| zg_norm_mean | 22.95 | 27.62 | 38.73 | 43.63 | 44.82 | 148 | 44.74 ± 0.05163 |
| zg_rel_spread | 0.0113 | 0.04145 | 0.03284 | 0.03053 | 0.0298 | 121 | 0.02986 ± 4.58e-05 |
| zg_n1_norm_mean | 23.04 | 27.62 | 38.73 | 43.67 | 44.85 | 148 | 44.77 ± 0.05352 |
| zg_n1_rel_spread | 0.01286 | 0.0415 | 0.03292 | 0.03058 | 0.02986 | 121 | 0.02992 ± 4.438e-05 |
| zv_pair_ratio | 0.3606 | 0.1146 | 0.1231 | 0.1283 | 0.1346 | 102 | 0.1339 ± 0.000499 |
| zv_pr | 1.067 | 4.74 | 5.192 | 5.433 | 5.577 | 135 | 5.565 ± 0.009064 |
| zg_pr | 1.009 | 1.338 | 1.433 | 1.508 | 1.554 | 150 | 1.55 ± 0.003835 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 39 | - | 3 | 0.999999 | 7 |
| zg | 51 | 182 (19pt) | 2 | 1 | 6 |
| zg_n1 | 51 | 182 (19pt) | 3 | 0.999999 | 6 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 68 | 141 | 150 | 155 | 155 |
| zg settle | 93 | 148 | 134 | 110 | 130 |
| zg_n1 settle | 93 | 148 | 134 | 110 | 130 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 8.833 | 0.2278 | 5.536 | 0.195 | 1.119 | 0.1743 | 40 |
| 2 | 8.767 | 0.2291 | 5.416 | 0.1614 | 1.126 | 0.1434 | 40 |
| 3 | 8.779 | 0.2295 | 5.45 | 0.1469 | 1.131 | 0.13 | 39 |
| 4 | 8.783 | 0.2306 | 5.445 | 0.1401 | 1.132 | 0.1237 | 39 |
| 5 | 8.836 | 0.2354 | 5.471 | 0.1386 | 1.136 | 0.122 | 39 |
| 6 | 8.885 | 0.2342 | 5.659 | 0.1411 | 1.139 | 0.1239 | 39 |
| 7 | 8.882 | 0.2417 | 5.54 | 0.1379 | 1.138 | 0.1212 | 39 |
| 8 | 8.88 | 0.2471 | 5.489 | 0.1425 | 1.139 | 0.1252 | 39 |
| 9 | 8.921 | 0.2472 | 5.618 | 0.1478 | 1.14 | 0.1296 | 39 |
| 10 | 8.88 | 0.2426 | 5.481 | 0.1557 | 1.138 | 0.1369 | 40 |
| 11 | 8.927 | 0.237 | 5.556 | 0.1716 | 1.138 | 0.1508 | 41 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.015 | 0.9997 | 1.467 | 0.9997 | 0.9997 | 0.9997 |
| 50 | 0.005853 | 0.9998 | 1.596 | 0.5914 | 0.5913 | 0.6009 |
| 100 | 0.003956 | 0.9996 | 1.696 | 0.5108 | 0.511 | 0.5185 |
| 150 | 0.00406 | 0.9993 | 1.726 | 0.5037 | 0.5043 | 0.5103 |
| 200 | 0.004571 | 0.9988 | 1.73 | 0.4961 | 0.4975 | 0.5044 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 49 | 0.015 | 0.004571 | 0.9988 |
| n1_vs_zv_az0 | 127 | 1.467 | 1.73 | 0.4961 |
| full_vs_zv_az0 | 126 | 1.467 | 1.73 | 0.4975 |
| full_vs_mean_pool | 127 | 1.466 | 1.732 | 0.5044 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.000 | 50 |
| 100 | 0.100 | 50 |
| 150 | 0.120 | 50 |
| 200 | 0.220 | 50 |

behaviour settles at epoch not_settled (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
