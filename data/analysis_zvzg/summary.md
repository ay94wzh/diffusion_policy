# latent series — /data/zihan/runs/run_square_zvzg_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 1.16e-04 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.0118, zg 0.042, zg_n1 0.0416 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 5.5e-09, zg 1.5e-09, zg_n1 1.3e-09 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.05, proc 1.27 | read against |
| chance level zg (unrelated) | rel 1.42, cos 1.00, cka 0.18, proc 1.30 | read against |
| chance level zg_n1 (unrelated) | rel 1.41, cos 1.00, cka 0.18, proc 1.28 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 33.9 | 21.81 | 17.7 | 17.06 | 17.1 | 94 | 17.09 ± 0.003964 |
| zv_rel_spread_mean | 0.03116 | 0.1204 | 0.1412 | 0.1486 | 0.1493 | 109 | 0.1492 ± 6.612e-05 |
| zv_rel_spread_min | 0.02543 | 0.113 | 0.1335 | 0.1402 | 0.1401 | 109 | 0.1401 ± 1.632e-05 |
| zg_norm_mean | 22.3 | 31.39 | 34.39 | 37.17 | 37.94 | 149 | 37.89 ± 0.03906 |
| zg_rel_spread | 0.01258 | 0.07511 | 0.08818 | 0.08713 | 0.08591 | 67 | 0.08601 ± 7.187e-05 |
| zg_n1_norm_mean | 22.29 | 28.89 | 32 | 34.59 | 35.36 | 146 | 35.31 ± 0.03526 |
| zg_n1_rel_spread | 0.01174 | 0.08285 | 0.09543 | 0.09407 | 0.09257 | 63 | 0.09267 ± 7.285e-05 |
| zv_pair_ratio | 0.4946 | 0.8513 | 0.781 | 0.7584 | 0.7506 | 107 | 0.7511 ± 0.0004268 |
| zv_pr | 1.976 | 11.02 | 11.1 | 11.05 | 10.92 | 12 | 10.93 ± 0.01061 |
| zg_pr | 1.91 | 5.461 | 5.497 | 5.454 | 5.419 | 22 | 5.424 ± 0.003595 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 23 | - | 5 | 1 | 10 |
| zg | 31 | 193 (8pt) | 4 | 1 | 11 |
| zg_n1 | 30 | 193 (8pt) | 3 | 1 | 13 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 73 | 130 | 138 | 100 | 146 |
| zg settle | 92 | 140 | 139 | 108 | 141 |
| zg_n1 settle | 47 | 138 | 137 | 109 | 140 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 17.16 | 0.1495 | 5.75 | 0.7804 | 1.011 | 0.772 | 23 |
| 2 | 17.8 | 0.1465 | 5.95 | 0.8037 | 0.9794 | 0.8206 | 21 |
| 3 | 17.7 | 0.1486 | 6.01 | 0.7887 | 0.9868 | 0.7992 | 17 |
| 4 | 16.75 | 0.1569 | 5.967 | 0.7625 | 1.036 | 0.736 | 20 |
| 5 | 17.71 | 0.1485 | 5.875 | 0.7657 | 0.9894 | 0.7739 | 19 |
| 6 | 17.24 | 0.1503 | 5.752 | 0.7478 | 1.02 | 0.7328 | 21 |
| 7 | 16.16 | 0.1587 | 5.717 | 0.6979 | 1.083 | 0.6446 | 28 |
| 8 | 16.28 | 0.1546 | 5.682 | 0.7001 | 1.071 | 0.6538 | 28 |
| 9 | 17.43 | 0.1401 | 5.743 | 0.7916 | 1.003 | 0.7889 | 25 |
| 10 | 17.14 | 0.1458 | 5.64 | 0.7783 | 1.019 | 0.7635 | 28 |
| 11 | 16.69 | 0.1433 | 5.489 | 0.805 | 1.021 | 0.7882 | 27 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.04646 | 0.9707 | 1.503 | 0.9739 | 0.968 | 0.9827 |
| 50 | 0.5745 | 0.9972 | 1.579 | 0.8453 | 0.8468 | 0.8372 |
| 100 | 0.4937 | 0.9975 | 1.574 | 0.8277 | 0.8308 | 0.8231 |
| 150 | 0.4459 | 0.9982 | 1.584 | 0.8149 | 0.8172 | 0.8131 |
| 200 | 0.4317 | 0.9984 | 1.586 | 0.8066 | 0.8089 | 0.8042 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 139 | 0.04646 | 0.4317 | 0.9984 |
| n1_vs_zv_az0 | 134 | 1.503 | 1.586 | 0.8066 |
| full_vs_zv_az0 | 142 | 1.504 | 1.593 | 0.8089 |
| full_vs_mean_pool | 126 | 1.503 | 1.643 | 0.8042 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.680 | 50 |
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
