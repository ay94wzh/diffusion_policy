# latent series — /data/zihan/runs/run_square_za_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 3.09e-04 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.00976, zg 0.0505, zg_n1 0.0521 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 4e-09, zg 4.3e-09, zg_n1 5.3e-10 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.27 | read against |
| chance level zg (unrelated) | rel 1.42, cos 1.01, cka 0.19, proc 1.29 | read against |
| chance level zg_n1 (unrelated) | rel 1.42, cos 1.01, cka 0.20, proc 1.28 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 33.75 | 10.32 | 7.426 | 6.961 | 6.994 | 72 | 6.986 ± 0.00571 |
| zv_rel_spread_mean | 0.01261 | 0.1491 | 0.1794 | 0.1821 | 0.1815 | 75 | 0.1816 ± 5.923e-05 |
| zv_rel_spread_min | 0.0102 | 0.1469 | 0.1717 | 0.1721 | 0.172 | 65 | 0.1721 ± 5.474e-05 |
| zg_norm_mean | 14.47 | 5.83 | 5.425 | 5.38 | 5.407 | 50 | 5.401 ± 0.004527 |
| zg_rel_spread | 0.007942 | 0.0925 | 0.1014 | 0.1047 | 0.1045 | 86 | 0.1046 ± 9.742e-05 |
| zg_n1_norm_mean | 14.51 | 5.828 | 5.381 | 5.346 | 5.387 | 50 | 5.38 ± 0.005075 |
| zg_n1_rel_spread | 0.009046 | 0.09315 | 0.1022 | 0.1049 | 0.1045 | 81 | 0.1046 ± 0.0001065 |
| zv_pair_ratio | 0.4548 | 0.2084 | 0.2337 | 0.239 | 0.2396 | 86 | 0.2393 ± 0.0001908 |
| zv_pr | 1.209 | 5.707 | 6.556 | 6.879 | 7.033 | 122 | 7.016 ± 0.01319 |
| zg_pr | 1.09 | 6.166 | 5.918 | 5.928 | 6.004 | 45 | 5.993 ± 0.008305 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 76 | - | 3 | 0.999999 | 7 |
| zg | 60 | 198 (3pt) | 3 | 1 | 7 |
| zg_n1 | 63 | 199 (2pt) | 4 | 1 | 8 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 74 | 143 | 151 | 157 | 157 |
| zg settle | 53 | 100 | 108 | 80 | 115 |
| zg_n1 settle | 54 | 102 | 112 | 82 | 123 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 7.214 | 0.172 | 6.786 | 0.389 | 1.168 | 0.333 | 91 |
| 2 | 7.036 | 0.1749 | 6.748 | 0.3189 | 1.198 | 0.2661 | 72 |
| 3 | 6.962 | 0.1774 | 6.851 | 0.2804 | 1.216 | 0.2307 | 73 |
| 4 | 6.918 | 0.1793 | 6.813 | 0.2655 | 1.226 | 0.2165 | 75 |
| 5 | 6.905 | 0.1809 | 6.719 | 0.2576 | 1.233 | 0.209 | 75 |
| 6 | 6.929 | 0.1815 | 6.674 | 0.2617 | 1.236 | 0.2117 | 78 |
| 7 | 6.935 | 0.1845 | 6.526 | 0.2641 | 1.238 | 0.2133 | 79 |
| 8 | 6.95 | 0.1839 | 6.607 | 0.2705 | 1.239 | 0.2184 | 77 |
| 9 | 6.988 | 0.1842 | 6.567 | 0.2817 | 1.237 | 0.2277 | 79 |
| 10 | 7.01 | 0.1862 | 6.563 | 0.2996 | 1.234 | 0.2427 | 80 |
| 11 | 7.085 | 0.1922 | 6.577 | 0.334 | 1.229 | 0.2719 | 83 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.01857 | 0.9994 | 1.53 | 0.9992 | 0.9992 | 0.9993 |
| 50 | 0.08981 | 0.9979 | 1.517 | 0.7531 | 0.7474 | 0.7536 |
| 100 | 0.07429 | 0.9984 | 1.471 | 0.694 | 0.6905 | 0.6975 |
| 150 | 0.06706 | 0.9987 | 1.462 | 0.6903 | 0.6887 | 0.6953 |
| 200 | 0.06479 | 0.9988 | 1.463 | 0.6883 | 0.6862 | 0.6921 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 150 | 0.01857 | 0.06479 | 0.9988 |
| n1_vs_zv_az0 | 103 | 1.53 | 1.463 | 0.6883 |
| full_vs_zv_az0 | 100 | 1.531 | 1.462 | 0.6862 |
| full_vs_mean_pool | 97 | 1.53 | 1.464 | 0.6921 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.600 | 50 |
| 100 | 0.760 | 50 |
| 150 | 0.760 | 50 |
| 200 | 0.820 | 50 |

behaviour settles at epoch not_settled (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
