# latent series — /data/zihan/runs/run_square_m4clear_s42_200ep/latent_snapshots

201 epochs analyzed (0..200), S=128, K=11, D=512, pool_ring=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], az0_pos=5
settle convention: within tol=0.05 x C of the final value (C = the curve's total excursion from it; settled also needs the last max(3, 10%) epochs inside the band); frozen floor=4.88e-04 relative

## gates

| gate | value | verdict |
|---|---|---|
| layout/meta/shape/finite | 201 analyzed, 0 skipped, holes=[] | PASS |
| state_idx identity | first mismatch: None | PASS |
| scalar recheck | max rel 8.05e-05 (tol 5e-03, 2010 pairs) | PASS |
| cka permutation floor | zv_flat 0.01, zg 0.05, zg_n1 0.05 (void at 0.9) | PASS |
| cka half-split gap | 0.000 | - |
| chance level zv_flat (unrelated) | rel 1.41, cos 1.00, cka 0.04, proc 1.28 | read against |
| chance level zg (unrelated) | rel 1.42, cos 1.01, cka 0.19, proc 1.24 | read against |
| chance level zg_n1 (unrelated) | rel 1.42, cos 1.01, cka 0.20, proc 1.21 | read against |
| fp16 storage floor | 4.9e-04 relative | convention |

## scalars — values at key epochs; settle at tol 0.05

| scalar | e0 | e50 | e100 | e150 | e200 | settle | tail mean ± std |
|---|---|---|---|---|---|---|---|
| zv_norm_mean | 31.92 | 10.16 | 7.293 | 6.85 | 6.91 | 75 | 6.9 ± 0.007695 |
| zv_rel_spread_mean | 0.04034 | 0.1958 | 0.2258 | 0.228 | 0.224 | 72 | 0.2246 ± 0.0004373 |
| zv_rel_spread_min | 0.03577 | 0.1708 | 0.1986 | 0.2043 | 0.2049 | 87 | 0.2053 ± 0.0003161 |
| zg_norm_mean | 16.8 | 4.102 | 3.536 | 3.449 | 3.47 | 49 | 3.466 ± 0.003455 |
| zg_rel_spread | 0.01545 | 0.1453 | 0.1412 | 0.1385 | 0.1372 | 75 | 0.1373 ± 0.0001014 |
| zg_n1_norm_mean | 17.24 | 4.071 | 3.486 | 3.421 | 3.45 | 48 | 3.446 ± 0.003035 |
| zg_n1_rel_spread | 0.01755 | 0.1421 | 0.1394 | 0.1359 | 0.1345 | 96 | 0.1345 ± 1.876e-05 |
| zv_pair_ratio | 0.857 | 0.5081 | 0.5089 | 0.4949 | 0.4846 | 126 | 0.4854 ± 0.0006981 |
| zv_pr | 2.066 | 5.546 | 6.063 | 5.998 | 5.822 | 149 | 5.837 ± 0.01205 |
| zg_pr | 1.927 | 5.664 | 5.841 | 5.865 | 5.873 | 55 | 5.872 ± 0.00163 |

## representation drift (epoch vs previous selected epoch)

| target | rel settle | rel frozen | cka settle | cka tail | proc settle |
|---|---|---|---|---|---|
| zv_flat | 53 | None | 3 | 1 | 12 |
| zg | 65 | 200 | 3 | 1 | 8 |
| zg_n1 | 70 | None | 4 | 1 | 21 |

drift vs the fixed refs (rel):

| target | vs e0 | vs e50 | vs e100 | vs e150 | vs e200 |
|---|---|---|---|---|---|
| zv_flat settle | 76 | 136 | 146 | 154 | 154 |
| zg settle | 55 | 104 | 108 | 85 | 115 |
| zg_n1 settle | 55 | 104 | 112 | 87 | 125 |

## per-view (final analyzed epoch) + settle of per-view rel_prev

| ring | norm | rel_spread | PR | pair_mean | cross_mean | ratio | rel_prev settle |
|---|---|---|---|---|---|---|---|
| 1 | 8.092 | 0.2204 | 4.224 | 0.7641 | 1.081 | 0.7072 | 44 |
| 2 | 7.348 | 0.2144 | 4.347 | 0.6438 | 1.14 | 0.5646 | 47 |
| 3 | 6.87 | 0.2229 | 4.456 | 0.5629 | 1.186 | 0.4745 | 52 |
| 4 | 6.587 | 0.2188 | 4.373 | 0.5161 | 1.219 | 0.4234 | 54 |
| 5 | 6.489 | 0.209 | 4.399 | 0.4919 | 1.244 | 0.3955 | 56 |
| 6 | 6.43 | 0.2049 | 4.336 | 0.4867 | 1.256 | 0.3876 | 57 |
| 7 | 6.483 | 0.2108 | 4.208 | 0.4921 | 1.257 | 0.3913 | 58 |
| 8 | 6.576 | 0.2251 | 4.248 | 0.5163 | 1.254 | 0.4118 | 58 |
| 9 | 6.705 | 0.2409 | 4.308 | 0.559 | 1.236 | 0.4524 | 55 |
| 10 | 6.98 | 0.2502 | 4.252 | 0.633 | 1.202 | 0.5266 | 51 |
| 11 | 7.449 | 0.2463 | 4.26 | 0.743 | 1.151 | 0.6453 | 48 |

## fused latents (M5 premise)

| epoch | rel(n1,full) | cka(n1,full) | rel(n1,zv_az0) | cka(n1,zv_az0) | cka(full,zv_az0) | cka(full,mean_pool) |
|---|---|---|---|---|---|---|
| 0 | 0.06398 | 0.9932 | 1.493 | 0.9657 | 0.9671 | 0.9663 |
| 50 | 0.1532 | 0.9962 | 1.568 | 0.7736 | 0.7665 | 0.7446 |
| 100 | 0.1207 | 0.9968 | 1.506 | 0.7255 | 0.7239 | 0.7068 |
| 150 | 0.1067 | 0.9975 | 1.493 | 0.7031 | 0.7015 | 0.6878 |
| 200 | 0.1031 | 0.9976 | 1.493 | 0.6913 | 0.6895 | 0.677 |

| fused relation | rel settle | rel e0 | rel e-final | cka e-final |
|---|---|---|---|---|
| n1_vs_full | 148 | 0.06398 | 0.1031 | 0.9976 |
| n1_vs_zv_az0 | 113 | 1.493 | 1.493 | 0.6913 |
| full_vs_zv_az0 | 110 | 1.499 | 1.493 | 0.6895 |
| full_vs_mean_pool | 115 | 1.49 | 1.49 | 0.677 |

## behaviour (from the log)

| epoch | test/mean_score | episodes |
|---|---|---|
| 0 | 0.000 | 50 |
| 50 | 0.440 | 50 |
| 100 | 0.700 | 50 |
| 150 | 0.800 | 50 |
| 200 | 0.760 | 50 |

behaviour settles at epoch not_settled (tol 0.05)

## caveats

- npz are fp16: differences below ~4.9e-04 relative
  are storage, not signal; deeper collapses can only be bounded by the npz
- 128 fixed states, one draw per condition, single seed, no held-out views:
  this is an instrument, not a ladder cell
- zg-vs-zv distances cross two latent spaces: read trends, not absolute values
- the four drift measures are read together; none alone is the answer
