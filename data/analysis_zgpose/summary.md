# latent series — /data/zihan/runs/run_square_zgpose_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 4.14e-04 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor (max) | zv_flat 0.00616, zg 0.0243, zg_n1 0.0242 (void at 0.9) | PASS |
| cka split-half gap (final pair) | zv_flat 1.3e-10, zg 2.8e-09, zg_n1 8.6e-09 | finite-sample floor |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.33 | read against |
| chance level zg (unrelated) | rel 1.42, cos 1.00, cka 0.13, proc 1.76 | read against |
| chance level zg_n1 (unrelated) | rel 1.42, cos 1.00, cka 0.12, proc 1.76 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 31.15 | 18.7 | 13.75 | 12.61 | 12.55 | 102 | 12.55 ± 0.0006577 |
| zv_rel_spread_mean | 0.005404 | 0.06846 | 0.08566 | 0.09223 | 0.09302 | 121 | 0.093 ± 2.911e-05 |
| zv_rel_spread_min | 0.004649 | 0.05748 | 0.07117 | 0.07801 | 0.07805 | 125 | 0.07809 ± 3.109e-05 |
| zg_norm_mean | 20.91 | 40.55 | 55.98 | 63.05 | 65.75 | 154 | 65.55 ± 0.1521 |
| zg_rel_spread | 0.002782 | 0.03872 | 0.03188 | 0.02876 | 0.02772 | 144 | 0.02778 ± 5.566e-05 |
| zg_n1_norm_mean | 20.96 | 41.03 | 57.53 | 65.1 | 67.05 | 142 | 66.91 ± 0.1058 |
| zg_n1_rel_spread | 0.003049 | 0.04218 | 0.0325 | 0.0289 | 0.02798 | 141 | 0.02803 ± 4.322e-05 |
| zv_pair_ratio | 1.299 | 1.74 | 1.427 | 1.325 | 1.308 | 91 | 1.308 ± 0.0005305 |
| zv_pr | 12.85 | 7.877 | 8.11 | 7.9 | 7.762 | 117 | 7.772 ± 0.007919 |
| zg_pr | 2.158 | 2.566 | 2.679 | 2.657 | 2.633 | 128 | 2.635 ± 0.001296 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen (trailing pts) | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 26 | 192 (9pt) | 5 | 1 | 8 |
| zg | 23 | 192 (9pt) | 4 | 1 | 11 |
| zg_n1 | 31 | 189 (12pt) | 4 | 1 | 12 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 87 | 133 | 124 | 109 | 126 |
| zg settle | 113 | 159 | 143 | 107 | 136 |
| zg_n1 settle | 105 | 145 | 130 | 109 | 126 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 13.24 | 0.1219 | 2.38 | 0.875 | 0.6734 | 1.299 | 19 |
| 2 | 13.43 | 0.0938 | 2.446 | 0.8468 | 0.6526 | 1.298 | 20 |
| 3 | 13.47 | 0.08191 | 2.45 | 0.8492 | 0.632 | 1.344 | 22 |
| 4 | 13.09 | 0.07805 | 2.514 | 0.8286 | 0.65 | 1.275 | 21 |
| 5 | 13.11 | 0.07855 | 2.515 | 0.8493 | 0.6394 | 1.328 | 25 |
| 6 | 12.8 | 0.08416 | 2.504 | 0.8413 | 0.6523 | 1.29 | 26 |
| 7 | 12.27 | 0.08497 | 2.593 | 0.8414 | 0.6745 | 1.247 | 33 |
| 8 | 11.72 | 0.09258 | 2.626 | 0.8566 | 0.7049 | 1.215 | 32 |
| 9 | 11.65 | 0.09184 | 2.658 | 0.8905 | 0.7108 | 1.253 | 36 |
| 10 | 11.54 | 0.1142 | 2.65 | 0.9576 | 0.7218 | 1.327 | 36 |
| 11 | 11.77 | 0.1011 | 2.649 | 1.065 | 0.7059 | 1.509 | 30 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.03323 | 0.9347 | 1.506 | 0.9344 | 0.7785 | 0.9902 |
| 50 | 0.4005 | 0.9944 | 1.629 | 0.9971 | 0.9886 | 0.9943 |
| 100 | 0.2482 | 0.9969 | 1.717 | 0.9835 | 0.9826 | 0.9876 |
| 150 | 0.2091 | 0.9972 | 1.753 | 0.9807 | 0.9823 | 0.9858 |
| 200 | 0.198 | 0.9975 | 1.76 | 0.979 | 0.9812 | 0.9844 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 112 | 0.03323 | 0.198 | 0.9975 |
| n1_vs_zv_az0 | 134 | 1.506 | 1.76 | 0.979 |
| full_vs_zv_az0 | 141 | 1.508 | 1.757 | 0.9812 |
| full_vs_mean_pool | 120 | 1.506 | 1.82 | 0.9844 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.220 | 50 |
| 100 | 0.440 | 50 |
| 150 | 0.780 | 50 |
| 200 | 0.820 | 50 |

behaviour settles at epoch not_settled (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
