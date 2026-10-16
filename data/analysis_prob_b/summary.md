# latent series — /data/zihan/runs/run_square_prob_b_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 9.25e-05 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.00925, zg 0.0574, zg_n1 0.06 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 7.9e-09, zg 2.9e-09, zg_n1 1.5e-09 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.28 | read against |
| chance level zg (unrelated) | rel 1.42, cos 1.00, cka 0.20, proc 1.19 | read against |
| chance level zg_n1 (unrelated) | rel 1.42, cos 1.00, cka 0.21, proc 1.17 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 30.61 | 13.42 | 10.28 | 9.752 | 9.778 | 84 | 9.773 ± 0.00435 |
| zv_rel_spread_mean | 0.03132 | 0.1554 | 0.1993 | 0.2065 | 0.2064 | 98 | 0.2067 ± 0.0001843 |
| zv_rel_spread_min | 0.02715 | 0.1336 | 0.1634 | 0.1717 | 0.1744 | 115 | 0.1745 ± 5.711e-05 |
| zg_norm_mean | 16.59 | 6.446 | 5.677 | 5.555 | 5.589 | 62 | 5.588 ± 0.001416 |
| zg_rel_spread | 0.01776 | 0.1244 | 0.1388 | 0.1389 | 0.1393 | 62 | 0.1392 ± 3.105e-05 |
| zg_n1_norm_mean | 16.72 | 6.44 | 5.62 | 5.541 | 5.589 | 61 | 5.587 ± 0.002172 |
| zg_n1_rel_spread | 0.01611 | 0.1244 | 0.1402 | 0.1401 | 0.1402 | 63 | 0.1401 ± 1.864e-05 |
| zv_pair_ratio | 0.4969 | 0.2232 | 0.2423 | 0.245 | 0.2438 | 65 | 0.2438 ± 4.262e-05 |
| zv_pr | 1.975 | 5.463 | 6.097 | 6.226 | 6.225 | 89 | 6.224 ± 0.001211 |
| zg_pr | 1.814 | 6.829 | 6.837 | 6.784 | 6.769 | 18 | 6.767 ± 0.002014 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 40 | - | 3 | 0.999999 | 9 |
| zg | 67 | 199 (2pt) | 3 | 1 | 8 |
| zg_n1 | 65 | 200 (1pt) | 4 | 1 | 10 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 78 | 136 | 145 | 153 | 152 |
| zg settle | 64 | 110 | 115 | 89 | 124 |
| zg_n1 settle | 64 | 110 | 117 | 90 | 131 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 10.18 | 0.1744 | 5.833 | 0.3616 | 1.1 | 0.3285 | 41 |
| 2 | 9.926 | 0.189 | 5.878 | 0.3016 | 1.128 | 0.2673 | 41 |
| 3 | 9.788 | 0.1995 | 5.959 | 0.2681 | 1.147 | 0.2337 | 41 |
| 4 | 9.695 | 0.2127 | 5.89 | 0.2536 | 1.159 | 0.2188 | 40 |
| 5 | 9.698 | 0.2169 | 5.87 | 0.2509 | 1.165 | 0.2153 | 40 |
| 6 | 9.683 | 0.2136 | 5.923 | 0.2541 | 1.17 | 0.2173 | 40 |
| 7 | 9.653 | 0.2209 | 5.803 | 0.2546 | 1.171 | 0.2174 | 40 |
| 8 | 9.695 | 0.2182 | 5.898 | 0.2628 | 1.172 | 0.2242 | 39 |
| 9 | 9.715 | 0.2115 | 5.915 | 0.2704 | 1.17 | 0.2311 | 40 |
| 10 | 9.737 | 0.2096 | 5.91 | 0.2912 | 1.166 | 0.2496 | 41 |
| 11 | 9.789 | 0.204 | 5.87 | 0.3288 | 1.159 | 0.2837 | 41 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.06785 | 0.9712 | 1.499 | 0.9677 | 0.9617 | 0.9762 |
| 50 | 0.1246 | 0.9974 | 1.556 | 0.8215 | 0.8212 | 0.8202 |
| 100 | 0.1119 | 0.9979 | 1.516 | 0.7864 | 0.7865 | 0.7849 |
| 150 | 0.09761 | 0.9983 | 1.509 | 0.7736 | 0.7752 | 0.7775 |
| 200 | 0.09168 | 0.9984 | 1.509 | 0.7689 | 0.7699 | 0.772 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 175 | 0.06785 | 0.09168 | 0.9984 |
| n1_vs_zv_az0 | 106 | 1.499 | 1.509 | 0.7689 |
| full_vs_zv_az0 | 102 | 1.501 | 1.51 | 0.7699 |
| full_vs_mean_pool | 105 | 1.497 | 1.512 | 0.772 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.580 | 50 |
| 100 | 0.760 | 50 |
| 150 | 0.740 | 50 |
| 200 | 0.800 | 50 |

behaviour settles at epoch not_settled (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
