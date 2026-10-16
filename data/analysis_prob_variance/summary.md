# variance quality -- /data/zihan/runs/run_square_prob_c_s42_200ep/checkpoints/latest.ckpt

fusion=poe aux_input=mu_v n_states=256 seed=0 (full 11-view draw)

## R1 view spread -- 0.4593 (log e200 0.44650840759277344) -- informative

## R2 per-view (ring) -- mean lv vs mean error
| ring | az | n | mean lv | err aux | err state (cm) |
|---|---|---|---|---|---|
| 1 | -75.0 | 512 | -1.2109 | 0.00596 | 3.5507 |
| 2 | -60.0 | 512 | -1.2360 | 0.00575 | 3.5384 |
| 3 | -45.0 | 512 | -1.3006 | 0.00535 | 3.6638 |
| 4 | -30.0 | 512 | -1.2326 | 0.00531 | 3.8382 |
| 5 | -15.0 | 512 | -1.2877 | 0.00554 | 3.5496 |
| 6 | +0.0 | 512 | -1.2050 | 0.00547 | 3.6494 |
| 7 | +15.0 | 512 | -1.2398 | 0.00561 | 3.4886 |
| 8 | +30.0 | 512 | -1.2151 | 0.00496 | 3.5337 |
| 9 | +45.0 | 512 | -1.2472 | 0.00525 | 3.6457 |
| 10 | +60.0 | 512 | -1.2699 | 0.00525 | 3.5161 |
| 11 | +75.0 | 512 | -1.2270 | 0.00559 | 3.8218 |
- rho_lv_vs_err_aux: rho=+0.209 p=0.2693 -- weak (rho=+0.209 below threshold or floor)
- rho_lv_vs_err_state: rho=+0.164 p=0.3161 -- weak (rho=+0.164 below threshold or floor)

## R3 pair-level (5632 live pairs; the state reading on 1127 held-out rows)
- rho_lv_vs_err_aux: rho=+0.023 p=0.0447 shuffle95=0.027 -- weak (rho=+0.023 below threshold or floor)
- rho_lv_vs_err_state: rho=-0.166 p=1.0000 shuffle95=0.057 -- null/finding-against (rho=-0.166 <= 0)

## R4 fused
- n_test_pairs: 1127
- rho_lvg_vs_state_residual: rho=-0.242 p=1.0000 shuffle95=0.060 -- null/finding-against (rho=-0.242 <= 0)
- rho_lvg_vs_diffusion_loss: rho=-0.031 p=0.7038 shuffle95=0.124 -- null/finding-against (rho=-0.031 <= 0)

## R5 N-sensitivity
| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lvg | -1.24 | -2.19 | -2.69 | -3.02 | -3.27 | -3.47 | -3.64 | -3.78 | -3.90 | -4.01 | -4.11 |
| drift | 0.2476 | 0.1626 | 0.1204 | 0.0945 | 0.0763 | 0.0622 | 0.0508 | 0.0406 | 0.0308 | 0.0204 | 0.0000 |
lvg non-increasing: True  drift non-increasing: True
az_0 removal drift 0.020402533933520317 vs worst edge 0.020708613097667694 -> registered ordering False

## R6 overconfidence
frac(sigma_g < between-view sigma) = 0.338; median ratio 1.4837 (p10 0.4278, p90 5.1448)
w_max = 0.2201 (uniform 0.0909); double-count factor 2.2255 (1 = one expert dominates, sqrt(K) = all counted independent)
