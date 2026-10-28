# latent series — /data/zihan/runs/run_square_zc_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 2.68e-04 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.009, zg 0.0134, zg_n1 0.0135 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 1.4e-08, zg 4.7e-09, zg_n1 5.3e-09 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.29 | read against |
| chance level zg (unrelated) | rel 1.42, cos 1.00, cka 0.09, proc 1.66 | read against |
| chance level zg_n1 (unrelated) | rel 1.42, cos 1.00, cka 0.09, proc 1.65 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 35.47 | 9.45 | 7.419 | 6.965 | 7.006 | 65 | 6.995 ± 0.007956 |
| zv_rel_spread_mean | 0.01144 | 0.1831 | 0.2292 | 0.2288 | 0.226 | 74 | 0.2263 ± 0.0002502 |
| zv_rel_spread_min | 0.008885 | 0.1788 | 0.2226 | 0.2209 | 0.2181 | 72 | 0.2184 ± 0.0001921 |
| zg_norm_mean | 19.08 | 6.275 | 6.085 | 5.745 | 5.71 | 41 | 5.708 ± 0.001602 |
| zg_rel_spread | 0.005773 | 0.08203 | 0.08802 | 0.09673 | 0.09891 | 131 | 0.09871 ± 0.000165 |
| zg_n1_norm_mean | 19.12 | 6.276 | 6.082 | 5.742 | 5.708 | 41 | 5.707 ± 0.001868 |
| zg_n1_rel_spread | 0.006591 | 0.08231 | 0.08823 | 0.09693 | 0.09912 | 132 | 0.09891 ± 0.0001663 |
| zv_pair_ratio | 0.402 | 0.1153 | 0.1298 | 0.1425 | 0.1541 | 148 | 0.1527 ± 0.0009744 |
| zv_pr | 1.117 | 4.543 | 5.449 | 6.058 | 6.52 | 166 | 6.473 ± 0.03519 |
| zg_pr | 1.02 | 1.469 | 1.327 | 1.3 | 1.298 | 77 | 1.297 ± 0.0003024 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 65 | - | 4 | 0.999999 | 9 |
| zg | 34 | 190 (11pt) | 4 | 1 | 8 |
| zg_n1 | 34 | 190 (11pt) | 4 | 1 | 14 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 65 | 153 | 160 | 162 | 162 |
| zg settle | 51 | 122 | 118 | 106 | 116 |
| zg_n1 settle | 51 | 122 | 118 | 106 | 116 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 7.009 | 0.2181 | 6.414 | 0.2198 | 1.145 | 0.192 | 65 |
| 2 | 6.959 | 0.2194 | 6.322 | 0.185 | 1.15 | 0.1609 | 62 |
| 3 | 6.949 | 0.2227 | 6.332 | 0.1683 | 1.155 | 0.1457 | 65 |
| 4 | 6.945 | 0.221 | 6.435 | 0.1607 | 1.157 | 0.1388 | 64 |
| 5 | 6.962 | 0.2232 | 6.414 | 0.1594 | 1.159 | 0.1375 | 65 |
| 6 | 7.027 | 0.2281 | 6.474 | 0.1659 | 1.162 | 0.1428 | 65 |
| 7 | 7.021 | 0.2307 | 6.414 | 0.1642 | 1.163 | 0.1412 | 64 |
| 8 | 7.035 | 0.2335 | 6.372 | 0.169 | 1.163 | 0.1453 | 65 |
| 9 | 7.062 | 0.2275 | 6.521 | 0.176 | 1.166 | 0.151 | 65 |
| 10 | 7.044 | 0.2309 | 6.46 | 0.1876 | 1.164 | 0.1612 | 66 |
| 11 | 7.055 | 0.2311 | 6.481 | 0.2078 | 1.163 | 0.1787 | 69 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.01077 | 0.9996 | 1.496 | 0.9999 | 0.9998 | 1 |
| 50 | 0.01249 | 0.9999 | 1.486 | 0.4021 | 0.4005 | 0.4065 |
| 100 | 0.007841 | 0.9999 | 1.451 | 0.3404 | 0.3398 | 0.3468 |
| 150 | 0.007817 | 0.9999 | 1.447 | 0.3475 | 0.347 | 0.3554 |
| 200 | 0.008676 | 0.9999 | 1.448 | 0.3538 | 0.3532 | 0.3601 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 67 | 0.01077 | 0.008676 | 0.9999 |
| n1_vs_zv_az0 | 82 | 1.496 | 1.448 | 0.3538 |
| full_vs_zv_az0 | 82 | 1.496 | 1.448 | 0.3532 |
| full_vs_mean_pool | 85 | 1.496 | 1.449 | 0.3601 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.040 | 50 |
| 100 | 0.060 | 50 |
| 150 | 0.080 | 50 |
| 200 | 0.080 | 50 |

behaviour settles at epoch not_settled (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
