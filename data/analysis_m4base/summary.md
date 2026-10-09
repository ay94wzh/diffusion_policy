# latent series — /data/zihan/runs/run_square_m4base_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 9.37e-05 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.00902, zg 0.0538, zg_n1 0.0557 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 2e-08, zg 3.6e-10, zg_n1 1.3e-08 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.27 | read against |
| chance level zg (unrelated) | rel 1.42, cos 1.01, cka 0.20, proc 1.19 | read against |
| chance level zg_n1 (unrelated) | rel 1.42, cos 1.01, cka 0.20, proc 1.18 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 30.64 | 12.07 | 8.785 | 8.3 | 8.388 | 80 | 8.376 ± 0.009586 |
| zv_rel_spread_mean | 0.03287 | 0.1535 | 0.2075 | 0.2259 | 0.2351 | 148 | 0.2346 ± 0.0004719 |
| zv_rel_spread_min | 0.02854 | 0.1316 | 0.1615 | 0.1851 | 0.198 | 164 | 0.1968 ± 0.0009819 |
| zg_norm_mean | 16.48 | 6.293 | 5.522 | 5.394 | 5.408 | 63 | 5.404 ± 0.004054 |
| zg_rel_spread | 0.01873 | 0.1372 | 0.1529 | 0.154 | 0.1544 | 64 | 0.1545 ± 7.347e-05 |
| zg_n1_norm_mean | 16.62 | 6.22 | 5.459 | 5.362 | 5.399 | 60 | 5.393 ± 0.005116 |
| zg_n1_rel_spread | 0.01724 | 0.139 | 0.1548 | 0.1548 | 0.1547 | 61 | 0.1548 ± 0.0001127 |
| zv_pair_ratio | 0.4842 | 0.2225 | 0.2422 | 0.2424 | 0.2393 | 57 | 0.2394 ± 0.0001666 |
| zv_pr | 1.847 | 5.456 | 6.023 | 5.938 | 5.753 | 148 | 5.768 ± 0.01285 |
| zg_pr | 1.767 | 6.365 | 6.276 | 6.277 | 6.275 | 41 | 6.274 ± 0.002501 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 47 | - | 3 | 0.999999 | 9 |
| zg | 66 | 199 (2pt) | 3 | 1 | 8 |
| zg_n1 | 65 | 200 (1pt) | 4 | 1 | 9 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 80 | 139 | 148 | 154 | 154 |
| zg settle | 65 | 111 | 115 | 89 | 124 |
| zg_n1 settle | 64 | 111 | 118 | 89 | 131 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 8.71 | 0.198 | 5.623 | 0.388 | 1.192 | 0.3254 | 47 |
| 2 | 8.46 | 0.2126 | 5.578 | 0.3177 | 1.218 | 0.2609 | 46 |
| 3 | 8.341 | 0.2271 | 5.553 | 0.2819 | 1.233 | 0.2285 | 48 |
| 4 | 8.295 | 0.2417 | 5.44 | 0.2664 | 1.242 | 0.2145 | 48 |
| 5 | 8.288 | 0.2505 | 5.322 | 0.2638 | 1.245 | 0.2118 | 46 |
| 6 | 8.327 | 0.247 | 5.438 | 0.2681 | 1.248 | 0.2147 | 47 |
| 7 | 8.341 | 0.2481 | 5.431 | 0.2694 | 1.251 | 0.2154 | 47 |
| 8 | 8.349 | 0.2462 | 5.418 | 0.2756 | 1.251 | 0.2204 | 48 |
| 9 | 8.35 | 0.2447 | 5.429 | 0.2834 | 1.25 | 0.2268 | 48 |
| 10 | 8.387 | 0.2419 | 5.52 | 0.3041 | 1.248 | 0.2438 | 47 |
| 11 | 8.417 | 0.2289 | 5.47 | 0.3399 | 1.24 | 0.2741 | 50 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.06906 | 0.9713 | 1.5 | 0.9697 | 0.9609 | 0.9785 |
| 50 | 0.127 | 0.9977 | 1.53 | 0.8006 | 0.7998 | 0.799 |
| 100 | 0.1093 | 0.998 | 1.468 | 0.7336 | 0.7337 | 0.7358 |
| 150 | 0.09541 | 0.9984 | 1.458 | 0.6983 | 0.6999 | 0.7042 |
| 200 | 0.08892 | 0.9985 | 1.461 | 0.6782 | 0.6782 | 0.6832 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 173 | 0.06906 | 0.08892 | 0.9985 |
| n1_vs_zv_az0 | 101 | 1.5 | 1.461 | 0.6782 |
| full_vs_zv_az0 | 98 | 1.501 | 1.462 | 0.6782 |
| full_vs_mean_pool | 99 | 1.498 | 1.465 | 0.6832 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.640 | 50 |
| 100 | 0.740 | 50 |
| 150 | 0.840 | 50 |
| 200 | 0.900 | 50 |

behaviour settles at epoch not_settled (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
