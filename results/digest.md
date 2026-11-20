# Digest of numerical results
Generated from results/*_summary.json by experiments/make_digest.py.

## Validation (Figure 5.4 setting of the book)
| n | R2 (per-sample L_max) | mu | kappa | F-F* SGD @50 | SAGA @50 | GD @50 | passes to 1e-8 SGD | SAGA | GD |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 18.27 | 0.07277 | 251 | 0.000459 | 1.11e-16 | 0.09416 | nan | 13.63 | nan |
| 10000 | 20.13 | 0.008047 | 2501 | 0.00035 | 0 | 0.1442 | nan | 11.38 | nan |

## B3: tightness of Proposition 5.9 (n = 1000, d = 50, 5 seeds, textbook step)

All attribution windows and window-averaged ratios use the instrumentation records with V_t >= 1e-18 (floor set by the precision of eta*).

### Problems
| config | lam | R2 | kappa | L_F | kappa_F=L_F/mu | cov cond | local kappa(H*) | lmin(H*)/mu | min sigma' | mean sigma' | d_eff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kappa=10,alpha=0 | 0.02778 | 0.2778 | 10 | 0.03218 | 1.158 | 2.425 | 1.087 | 1.065 | 0.2382 | 0.2486 | 14.63 |
| kappa=10,alpha=1 | 0.02778 | 0.2778 | 10 | 0.04411 | 1.588 | 59.32 | 1.57 | 1.01 | 0.2418 | 0.2492 | 7.036 |
| kappa=10,alpha=2 | 0.02778 | 0.2778 | 10 | 0.04809 | 1.731 | 2798 | 1.699 | 1 | 0.2062 | 0.246 | 2.194 |
| kappa=100,alpha=0 | 0.002525 | 0.2525 | 100 | 0.006925 | 2.742 | 2.425 | 1.535 | 1.609 | 0.0621 | 0.2138 | 40.79 |
| kappa=100,alpha=1 | 0.002525 | 0.2525 | 100 | 0.01886 | 7.467 | 59.32 | 6.255 | 1.1 | 0.119 | 0.2299 | 27.62 |
| kappa=100,alpha=2 | 0.002525 | 0.2525 | 100 | 0.02283 | 9.042 | 2798 | 7.287 | 1.003 | 0.07339 | 0.2204 | 7.928 |
| kappa=1000,alpha=0 | 0.00025 | 0.2503 | 1000 | 0.00465 | 18.58 | 2.425 | 3.416 | 3.275 | 0.000867 | 0.1396 | 48.9 |
| kappa=1000,alpha=1 | 0.00025 | 0.2503 | 1000 | 0.01658 | 66.26 | 59.32 | 25.17 | 1.737 | 0.007395 | 0.1707 | 45.63 |
| kappa=1000,alpha=2 | 0.00025 | 0.2503 | 1000 | 0.02056 | 82.15 | 2798 | 49.5 | 1.02 | 0.006099 | 0.1739 | 21.79 |
| kappa=10000,alpha=0 | 2.5e-05 | 0.25 | 1e+04 | 0.004424 | 177 | 2.425 | 8.117 | 9.863 | 1.41e-05 | 0.1019 | 49.89 |
| kappa=10000,alpha=1 | 2.5e-05 | 0.25 | 1e+04 | 0.01636 | 654.2 | 59.32 | 60.33 | 4.97 | 8.9e-05 | 0.121 | 49.51 |
| kappa=10000,alpha=2 | 2.5e-05 | 0.25 | 1e+04 | 0.02033 | 813.2 | 2798 | 328.7 | 1.142 | 0.00037 | 0.1325 | 40.81 |
| kappa=100000,alpha=0 | 2.5e-06 | 0.25 | 1e+05 | 0.004402 | 1761 | 2.425 | 10.64 | 69.88 | 3.93e-06 | 0.09397 | 49.99 |
| kappa=100000,alpha=1 | 2.5e-06 | 0.25 | 1e+05 | 0.01633 | 6534 | 59.32 | 98.94 | 26.26 | 8.91e-06 | 0.1047 | 49.95 |
| kappa=100000,alpha=2 | 2.5e-06 | 0.25 | 1e+05 | 0.02031 | 8124 | 2798 | 1438 | 2.121 | 3.14e-05 | 0.1078 | 48.72 |
### Attribution of the log-gap (passes 2-20) and window-averaged ratios
| config | speed-up exact/bound | share I1 | I3 | I2 | I4 | I5 | r_sq | r_coco=L_eff/R2 | r_var | r_sc=mu/mu_eff | r_min | r_init | final E D_t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kappa=10,alpha=0 | 3.608 | 0.3947 | 0.004557 | 0.3526 | 0.02591 | 0.2222 | 0.5018 | 0.1648 | 0.9939 | 0.899 | 0.9998 | 0.01883 | 9.08e-27 |
| kappa=100,alpha=0 | 3.172 | 0.4428 | 0.000901 | 0.317 | 0.1314 | 0.1079 | 0.5206 | 0.2753 | 0.9989 | 0.5013 | 0.9999 | 0.005972 | 8.54e-27 |
| kappa=1000,alpha=0 | 5.11 | 0.3838 | 0.000976 | 0.2482 | 0.2044 | 0.1626 | 0.5203 | 0.3743 | 0.9986 | 0.1438 | 0.9999 | 0.002542 | 9.62e-27 |
| kappa=10000,alpha=0 | 29.46 | 0.1574 | 0.000663 | 0.4574 | 0.3117 | 0.07282 | 0.1384 | 0.1976 | 0.9956 | 0.08968 | 1 | 0.001372 | 5.14e-20 |
| kappa=100000,alpha=0 | 220.4 | 0.1413 | 0.000471 | 0.458 | 0.3406 | 0.0596 | 0.07537 | 0.2018 | 0.9964 | 0.01229 | 1 | 0.001167 | 1e-13 |
| kappa=10,alpha=1 | 3.578 | 0.4021 | 0.005375 | 0.3491 | 0.03214 | 0.2113 | 0.5008 | 0.1552 | 0.993 | 0.8707 | 0.9998 | 0.0158 | 1.26e-21 |
| kappa=100,alpha=1 | 3.335 | 0.4135 | 0.001378 | 0.3484 | 0.1359 | 0.1008 | 0.5118 | 0.1628 | 0.9983 | 0.5225 | 0.9999 | 0.00273 | 9.17e-27 |
| kappa=1000,alpha=1 | 5.155 | 0.2105 | 0.001849 | 0.5137 | 0.204 | 0.07009 | 0.3835 | 0.1289 | 0.9936 | 0.4233 | 0.9999 | 0.000851 | 5.53e-25 |
| kappa=10000,alpha=1 | 15.73 | 0.07634 | 0.000277 | 0.5756 | 0.3234 | 0.02439 | 0.03232 | 0.1134 | 0.9949 | 0.1717 | 1 | 0.000334 | 7.11e-09 |
| kappa=100000,alpha=1 | 100.3 | 0.07194 | 0.000165 | 0.544 | 0.3617 | 0.0222 | 0.01521 | 0.1211 | 0.9967 | 0.02724 | 1 | 0.000234 | 0.001003 |
| kappa=10,alpha=2 | 3.71 | 0.4035 | 0.005173 | 0.3396 | 0.0788 | 0.1729 | 0.5005 | 0.1821 | 0.9933 | 0.7377 | 0.9998 | 0.02793 | 7.88e-27 |
| kappa=100,alpha=2 | 3.518 | 0.3923 | 0.002944 | 0.3651 | 0.1688 | 0.0708 | 0.5031 | 0.09803 | 0.9961 | 0.4332 | 0.9999 | 0.003274 | 8.51e-27 |
| kappa=1000,alpha=2 | 3.366 | 0.04883 | 0.001518 | 0.8284 | 0.1055 | 0.01572 | 0.2111 | 0.01914 | 0.9581 | 0.7936 | 1 | 0.000544 | 1.18e-22 |
| kappa=10000,alpha=2 | 5.968 | 0.03019 | 0.000175 | 0.7138 | 0.2468 | 0.009037 | 0.01231 | 0.04194 | 0.9928 | 0.4564 | 1 | 0.000114 | 0.4877 |
| kappa=100000,alpha=2 | 18.95 | 0.03296 | 7e-05 | 0.619 | 0.3388 | 0.009096 | 0.003488 | 0.0586 | 0.9972 | 0.1443 | 1 | 3.13e-05 | 878.3 |
### Attribution in order A of the B3 note (undo I4, I2, I3, I1, then the min I5; passes 2-20)
| config | share I4 | I2 | I3 | I1 | I5 (min, undone last) |
|---|---|---|---|---|---|
| kappa=10,alpha=0 | 0 | 0 | 0.004638 | 0.3846 | 0.6108 |
| kappa=100,alpha=0 | 0 | 0 | 0.001003 | 0.4418 | 0.5572 |
| kappa=1000,alpha=0 | 0.1893 | 0 | 0.001175 | 0.4134 | 0.3961 |
| kappa=10000,alpha=0 | 0.3607 | 0.2291 | 0.005464 | 0.3362 | 0.06857 |
| kappa=100000,alpha=0 | 0.3716 | 0.4167 | 0.002803 | 0.1244 | 0.0845 |
| kappa=10,alpha=1 | 0 | 0 | 0.005415 | 0.3927 | 0.6019 |
| kappa=100,alpha=1 | 0 | 0 | 0.001491 | 0.4184 | 0.5801 |
| kappa=1000,alpha=1 | 0.1872 | 0 | 0.005441 | 0.5218 | 0.2856 |
| kappa=10000,alpha=1 | 0.3311 | 0.5785 | 0.001621 | 0.03769 | 0.05117 |
| kappa=100000,alpha=1 | 0.3667 | 0.5515 | 0 | 0.0296 | 0.05214 |
| kappa=10,alpha=2 | 0 | 0 | 0.005208 | 0.3888 | 0.606 |
| kappa=100,alpha=2 | 0 | 0 | 0.003064 | 0.3952 | 0.6017 |
| kappa=1000,alpha=2 | 0.113 | 0.2158 | 0.06256 | 0.5828 | 0.02586 |
| kappa=10000,alpha=2 | 0.2476 | 0.716 | 0 | 0.01234 | 0.02407 |
| kappa=100000,alpha=2 | 0.3392 | 0.6197 | 0 | 0.01514 | 0.02602 |
### Late window (passes 20-100) and leave-one-out exponents (per iteration, passes 2-20)

loo columns: -log of the single-substitution rates of vrlab.analysis.leave_one_out (one inequality kept measured, the others applied, no max), averaged over the window. They are not nested in the proof chain: 'loo I1' and 'loo I2' are hybrids whose rate can exceed 1, hence the negative values; they are diagnostics, not contraction exponents.

| config | speed-up late | late I1 | I3 | I2 | I4 | I5 | c_exact | c_bound | loo I1 | loo I3 | loo I2 | loo I4 | loo I5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kappa=10,alpha=0 | 3.539 | 0.4149 | 2.44e-05 | 0.3436 | 0.02456 | 0.2169 | 0.001203 | 0.000333 | 0.000956 | 0.000532 | 0.000799 | 0.000551 | 0.000528 |
| kappa=100,alpha=0 | 3.171 | 0.4419 | 0.000157 | 0.3171 | 0.133 | 0.1079 | 0.001058 | 0.000333 | 0.000665 | 0.000412 | 0.000483 | 0.000507 | 0.000412 |
| kappa=1000,alpha=0 | 5.071 | 0.3781 | 0.000428 | 0.2449 | 0.2105 | 0.166 | 0.000958 | 0.000188 | 0.000423 | 0.000314 | 0.000241 | 0.00047 | 0.000313 |
| kappa=10000,alpha=0 | 26.33 | 0.1251 | 0.000431 | 0.5009 | 0.3312 | 0.04237 | 0.000552 | 1.88e-05 | -3.76e-05 | 5.8e-05 | 2.44e-05 | 0.000224 | 5.76e-05 |
| kappa=100000,alpha=0 | 186.6 | 0.1152 | 0.000251 | 0.4899 | 0.3588 | 0.03582 | 0.000413 | 1.88e-06 | -7.35e-05 | 2.66e-05 | -1.88e-05 | 0.000167 | 2.64e-05 |
| kappa=10,alpha=1 | 3.519 | 0.4298 | 2.31e-05 | 0.3338 | 0.03624 | 0.2001 | 0.001193 | 0.000333 | 0.000935 | 0.000522 | 0.000774 | 0.000546 | 0.000517 |
| kappa=100,alpha=1 | 3.428 | 0.4009 | 0.000153 | 0.3517 | 0.1507 | 0.09657 | 0.001112 | 0.000333 | 0.000666 | 0.000413 | 0.000507 | 0.000518 | 0.000412 |
| kappa=1000,alpha=1 | 4.817 | 0.168 | 0.001592 | 0.584 | 0.1977 | 0.04866 | 0.000967 | 0.000188 | 0.000303 | 0.000244 | 0.000377 | 0.000401 | 0.000242 |
| kappa=10000,alpha=1 | 13.3 | 0.06614 | 0.000174 | 0.5954 | 0.3201 | 0.01811 | 0.000295 | 1.88e-05 | -3.87e-05 | 2.56e-05 | 3.56e-05 | 0.000115 | 2.55e-05 |
| kappa=100000,alpha=1 | 71.64 | 0.06208 | 7.83e-05 | 0.5575 | 0.3637 | 0.01665 | 0.000188 | 1.88e-06 | -5.23e-05 | 6.04e-06 | -4.95e-06 | 7.34e-05 | 6.01e-06 |
| kappa=10,alpha=2 | 3.553 | 0.4206 | 1.86e-05 | 0.3309 | 0.08262 | 0.1659 | 0.001237 | 0.000333 | 0.000844 | 0.000498 | 0.000684 | 0.00057 | 0.000494 |
| kappa=100,alpha=2 | 3.719 | 0.3667 | 0.000102 | 0.3774 | 0.1939 | 0.06188 | 0.001173 | 0.000333 | 0.000607 | 0.000395 | 0.000461 | 0.000535 | 0.000392 |
| kappa=1000,alpha=2 | 2.864 | 0.008298 | 0.000776 | 0.9496 | 0.03982 | 0.001526 | 0.000631 | 0.000188 | 0.000286 | 0.000195 | 0.000484 | 0.000241 | 0.000194 |
| kappa=10000,alpha=2 | 4 | 0.02029 | 5.69e-05 | 0.8082 | 0.1665 | 0.005025 | 0.000112 | 1.88e-05 | 1.07e-05 | 1.96e-05 | 4.78e-05 | 4.26e-05 | 1.96e-05 |
| kappa=100000,alpha=2 | 11.29 | 0.03229 | 1.4e-05 | 0.6455 | 0.3141 | 0.008145 | 3.55e-05 | 1.88e-06 | -7.38e-06 | 2.18e-06 | 4.01e-06 | 1.36e-05 | 2.18e-06 |

## B2: effective curvature along trajectories (n = 1000, d = 50)
| config | lmin(H*)/mu | lmax(H*)/mu | SAGA mu_eff/mu start | geomean | end | GD mu_eff/mu end | SAGA L_eff/R2 geomean | GD L_eff/R2 end | L_eff limit min/R2 | max/R2 | SAGA energy on slowest eigvec (end) | GD (end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha=0,kappa=100 | 1.609 | 2.471 | 2.086 | 1.996 | 2.088 | 1.635 | 0.2759 | 0.2031 | 0.1903 | 0.3448 | 0.000821 | 0.5296 |
| alpha=0,kappa=1000 | 3.275 | 11.19 | 7.917 | 7.137 | 7.341 | 3.275 | 0.378 | 0.1743 | 0.1645 | 0.4753 | 0.01468 | 1 |
| alpha=0,kappa=10000 | 9.863 | 80.06 | 50.05 | 10.59 | 9.942 | 9.863 | 0.189 | 0.1808 | 0.1725 | 0.5257 | 0.9976 | 1 |
| alpha=1,kappa=100 | 1.1 | 6.881 | 1.793 | 1.901 | 2.147 | 1.155 | 0.1609 | 0.05002 | 0.03531 | 0.3706 | 0.000815 | 0.0744 |
| alpha=1,kappa=1000 | 1.737 | 43.71 | 4.632 | 2.198 | 1.904 | 1.9 | 0.1277 | 0.104 | 0.08695 | 0.343 | 0.3076 | 0.1477 |
| alpha=1,kappa=10000 | 4.97 | 299.8 | 21.02 | 5.386 | 4.974 | 5.058 | 0.1103 | 0.1074 | 0.08562 | 0.3404 | 0.9996 | 0.9958 |
| alpha=2,kappa=100 | 1.003 | 7.305 | 2.497 | 2.167 | 4.98 | 1.029 | 0.09093 | 0.01327 | 0.01027 | 0.2058 | 7.61e-06 | 0.000955 |
| alpha=2,kappa=1000 | 1.02 | 50.47 | 5.34 | 1.163 | 1.062 | 1.129 | 0.01161 | 0.01231 | 0.002567 | 0.2111 | 0.002485 | 0.000958 |
| alpha=2,kappa=10000 | 1.142 | 375.5 | 13.79 | 1.809 | 1.468 | 1.787 | 0.03506 | 0.03676 | 0.008682 | 0.223 | 0.001688 | 0.000715 |

Protocol (read from results/B2.pkl): theta_0 = 0; SAGA 60 passes, a record every 100 iterations; GD at most 400 iterations, a record every 1. Records with D_t = ||theta_t - eta*||^2 < 1e-18 are discarded and 'end' is the last record above this floor. pi_min: share of D_t on the eigenvector q_min of H* = F''(eta*) with the smallest eigenvalue lmin(H*); d_eff = 1/sum_k pi_k^2.

### B2: bottom of the spectrum of H*, criterion of Rem. B2-P2-outcome, initial state theta_0 = 0 (SAGA and GD)
| config | lambda'/lmin(H*) | # eigs of H* <= 1.1 lmin(H*) | 9n lmin(H*)/(16R2) | 9n mu/(16R2) = 9n/(16 kappa) | pi_min(theta_0) | d_eff(theta_0) | Rayleigh quotient of H* at theta_0 - eta*, /mu | R_x sqrt(D_0) |
|---|---|---|---|---|---|---|---|---|
| alpha=0,kappa=100 | 1.01841 | 10 | 9.053 | 5.625 | 0.03676 | 8.656 | 1.823 | 7.775 |
| alpha=0,kappa=1000 | 1.45795 | 1 | 1.842 | 0.5625 | 0.9566 | 1.093 | 3.43 | 20.33 |
| alpha=0,kappa=10000 | 2.67939 | 1 | 0.5548 | 0.05625 | 0.984 | 1.033 | 10.38 | 31.9 |
| alpha=1,kappa=100 | 1.00157 | 21 | 6.189 | 5.625 | 0.000726 | 13.24 | 1.679 | 6.58 |
| alpha=1,kappa=1000 | 1.00503 | 5 | 0.9769 | 0.5625 | 0.002954 | 10.51 | 2.796 | 22.05 |
| alpha=1,kappa=10000 | 1.2116 | 1 | 0.2796 | 0.05625 | 0.8879 | 1.267 | 5.885 | 44.21 |
| alpha=2,kappa=100 | 1.0001 | 42 | 5.639 | 5.625 | 7.82e-06 | 3.954 | 2.173 | 5.945 |
| alpha=2,kappa=1000 | 1.00086 | 29 | 0.5735 | 0.5625 | 3.05e-05 | 6.234 | 3.168 | 19.73 |
| alpha=2,kappa=10000 | 1.00731 | 11 | 0.06426 | 0.05625 | 0.000148 | 17.97 | 4.829 | 50.65 |
### B2: SAGA, end state and alignment with q_min (times in passes)
| config | end (passes) | first record below the floor | D_t (end) | R_x sqrt(D_t) (end) | min pi_min | at pass | first pass with pi_min >= 0.99 | pi_min at pass 10 | d_eff at pass 10 | d_eff (end) | d_eff geomean | share of the 5 lowest eigvecs (end) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha=0,kappa=100 | 39.5 | 38.3 | 1.06e-18 | 1.03e-09 | 1.13e-07 | 15.2 | never | 0.009489 | 22.19 | 23.67 | 17.76 | 0.06004 |
| alpha=0,kappa=1000 | 43.6 | 43.7 | 1.08e-18 | 1.04e-09 | 1.31e-07 | 32.8 | never | 0.02923 | 16.39 | 23.71 | 13.84 | 0.07692 |
| alpha=0,kappa=10000 | 59.9 | never | 2.16e-11 | 4.65e-06 | 0.9381 | 3.1 | 9.2 | 0.9933 | 1.014 | 1.005 | 1.013 | 0.9982 |
| alpha=1,kappa=100 | 35.8 | 35.9 | 1.19e-18 | 1.09e-09 | 1.15e-07 | 13.4 | never | 0.005764 | 16.66 | 15.16 | 13.43 | 0.0498 |
| alpha=1,kappa=1000 | 48.9 | 49 | 1.04e-18 | 1.02e-09 | 0.002954 | 0 | never | 0.01637 | 5.4 | 3.788 | 4.057 | 0.8836 |
| alpha=1,kappa=10000 | 59.9 | never | 0.000154 | 0.01243 | 0.8879 | 0 | 15.7 | 0.9804 | 1.04 | 1.001 | 1.021 | 0.9998 |
| alpha=2,kappa=100 | 34.2 | 33.4 | 1.84e-18 | 1.35e-09 | 4.27e-11 | 32.7 | never | 0.002512 | 5.944 | 2.708 | 5.177 | 0.000704 |
| alpha=2,kappa=1000 | 59.9 | never | 1.88e-13 | 4.34e-07 | 2.01e-05 | 0.5 | never | 0.000387 | 13.14 | 12.1 | 12.47 | 0.08974 |
| alpha=2,kappa=10000 | 59.9 | never | 8.57 | 2.927 | 0.000148 | 0 | never | 0.000448 | 17.44 | 11.9 | 14.21 | 0.02937 |
### B2: SAGA, effective curvature relative to mu and to lmin(H*)
| config | mu_eff/lmin(H*) (end) | quadratic proxy/mu (end) | (proxy - mu_eff)/mu_eff (end) | min mu_eff/mu (kept records) | min mu_eff/mu (all records) | frac. of non-increasing steps | mu_eff/mu 10 passes before the end | frac. non-increasing, last 10 passes | R2 / geomean L_eff |
|---|---|---|---|---|---|---|---|---|---|
| alpha=0,kappa=100 | 1.297 | 2.088 | -6.81e-06 | 1.882 | 1.877 | 0.5326 | 1.956 | 0.4659 | 3.624 |
| alpha=0,kappa=1000 | 2.242 | 7.341 | 1.66e-07 | 4.738 | 4.738 | 0.5413 | 7.315 | 0.54 | 2.645 |
| alpha=0,kappa=10000 | 1.008 | 9.942 | -1.54e-07 | 9.9 | 9.9 | 0.5042 | 9.972 | 0.46 | 5.291 |
| alpha=1,kappa=100 | 1.951 | 2.147 | 5.48e-08 | 1.382 | 1.382 | 0.5503 | 1.771 | 0.53 | 6.215 |
| alpha=1,kappa=1000 | 1.096 | 1.904 | -8.9e-05 | 1.857 | 1.802 | 0.5112 | 1.857 | 0.46 | 7.834 |
| alpha=1,kappa=10000 | 1.001 | 4.973 | -0.000196 | 4.973 | 4.973 | 0.7245 | 4.979 | 0.5 | 9.07 |
| alpha=2,kappa=100 | 4.967 | 4.98 | 3.4e-05 | 1.162 | 1.162 | 0.5237 | 4.454 | 0.5208 | 11 |
| alpha=2,kappa=1000 | 1.042 | 1.062 | 1.18e-08 | 1.061 | 1.061 | 0.7062 | 1.067 | 0.65 | 86.11 |
| alpha=2,kappa=10000 | 1.285 | 1.465 | -0.002044 | 1.468 | 1.468 | 0.995 | 1.501 | 1 | 28.53 |
### B2: SAGA over the last five recorded passes before the end, and decay of D_t over the second half

States above the floor with t >= t_end - 5n. Decay: minus the least-squares slope of log D_t against passes over the kept states with t >= t_end/2; slowest mode: 2 n gamma lmin(H*) = n lmin(H*)/(2 R2) per pass (textbook step).

| config | # states | mu_eff/mu geomean | min | max | mu_eff/lmin(H*) geomean | min | max | d_eff mean | min | max | max pi_min | mean share of the 5 lowest eigvecs | decay of D_t per pass (2nd half) | slowest mode per pass | decay / slowest mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha=0,kappa=100 | 39 | 2.009 | 1.935 | 2.095 | 1.248 | 1.202 | 1.301 | 18.22 | 10.52 | 26.36 | 0.04918 | 0.08691 | 1.107 | 8.047 | 0.1375 |
| alpha=0,kappa=1000 | 51 | 7.552 | 6.646 | 8.746 | 2.306 | 2.029 | 2.671 | 17.18 | 12.55 | 25.75 | 0.0338 | 0.08458 | 1.098 | 1.637 | 0.6704 |
| alpha=0,kappa=10000 | 51 | 9.939 | 9.9 | 9.976 | 1.008 | 1.004 | 1.011 | 1.004 | 1.002 | 1.006 | 0.999 | 0.9982 | 0.495 | 0.4931 | 1.004 |
| alpha=1,kappa=100 | 51 | 1.875 | 1.382 | 3.628 | 1.704 | 1.256 | 3.298 | 13.93 | 4.462 | 21.47 | 0.02635 | 0.03953 | 1.211 | 5.501 | 0.2201 |
| alpha=1,kappa=1000 | 51 | 2 | 1.857 | 3.962 | 1.152 | 1.069 | 2.281 | 3.768 | 3.606 | 4.181 | 0.3114 | 0.8903 | 0.9135 | 0.8684 | 1.052 |
| alpha=1,kappa=10000 | 51 | 4.976 | 4.973 | 4.979 | 1.001 | 1.001 | 1.002 | 1.001 | 1.001 | 1.001 | 0.9996 | 0.9998 | 0.2489 | 0.2485 | 1.001 |
| alpha=2,kappa=100 | 47 | 2.604 | 1.324 | 6.026 | 2.597 | 1.321 | 6.011 | 5.38 | 1.67 | 9.377 | 0.000784 | 0.005028 | 1.22 | 5.013 | 0.2433 |
| alpha=2,kappa=1000 | 51 | 1.063 | 1.061 | 1.064 | 1.043 | 1.041 | 1.044 | 12.07 | 12.02 | 12.1 | 0.002485 | 0.08549 | 0.5365 | 0.5098 | 1.053 |
| alpha=2,kappa=10000 | 51 | 1.475 | 1.468 | 1.483 | 1.291 | 1.285 | 1.298 | 11.96 | 11.9 | 12.03 | 0.001688 | 0.0283 | 0.07635 | 0.05712 | 1.337 |
### B2: gradient descent (step 1/L_F; times in iterations)
| config | end (iteration) | D_t (end) | share of the 5 lowest eigvecs (end) | first iteration with pi_min >= 0.99 | d_eff (end) | mu_eff/lmin(H*) (end) | min mu_eff/mu (kept records) | min mu_eff/mu (all records) | frac. of non-increasing steps |
|---|---|---|---|---|---|---|---|---|---|
| alpha=0,kappa=100 | 23 | 5.66e-18 | 0.986 | never | 2.512 | 1.016 | 1.635 | 1.309 | 1 |
| alpha=0,kappa=1000 | 118 | 1.09e-18 | 1 | 3 | 1 | 1 | 3.275 | 3.267 | 0.8898 |
| alpha=0,kappa=10000 | 400 | 1.51e-18 | 1 | 1 | 1 | 1 | 9.863 | 9.863 | 0.86 |
| alpha=1,kappa=100 | 127 | 1.13e-18 | 0.2228 | never | 10.06 | 1.05 | 1.155 | 1.14 | 1 |
| alpha=1,kappa=1000 | 400 | 4.7e-09 | 0.6584 | never | 3.89 | 1.094 | 1.9 | 1.9 | 1 |
| alpha=1,kappa=10000 | 400 | 1.018 | 0.9987 | 249 | 1.008 | 1.018 | 5.058 | 5.058 | 1 |
| alpha=2,kappa=100 | 171 | 1.09e-18 | 0.006052 | never | 11.64 | 1.026 | 1.029 | 0.7949 | 1 |
| alpha=2,kappa=1000 | 400 | 0.000568 | 0.01032 | never | 14.62 | 1.107 | 1.129 | 1.129 | 1 |
| alpha=2,kappa=10000 | 400 | 201.5 | 0.01228 | never | 15.34 | 1.565 | 1.787 | 1.787 | 1 |

## B1: preconditioned SAGA (n = 1000, d = 50, kappa = 1e3, 5 seeds)
| alpha|P | R2_P | mu_P global | mu_P local | kappa_P | kappa_P local (F'' vs P) | max lev | mean lev | emp rate/pass (D_P) | emp rate/pass (Fgap) | bound rate/pass (global mu_P) | bound (local mu_P) | r_coco | r_sc | r_sq | r_var | r_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| alpha=1|I | 0.2503 | 0.00025 | 0.000435 | 1000 | 25.17 | 1 | 0.3059 | 1 | 1.011 | 0.1875 | 0.3256 | 0.1289 | 0.4233 | 0.3835 | 0.9936 | 0.9999 |
| alpha=1|diag | 34.95 | 0.0184 | 0.05451 | 1899 | 25.67 | 139.5 | 47.84 | 0.9414 | 0.9829 | 0.09872 | 0.2924 | 0.1636 | 0.2108 | 0.399 | 0.9956 | 0.9999 |
| alpha=1|stat | 19.4 | 0.003816 | 0.1166 | 5083 | 2.801 | 76.84 | 45.63 | 1.114 | 1.109 | 0.03689 | 0.3333 | 0.3288 | 0.01692 | 0.5245 | 0.999 | 0.9997 |
| alpha=1|hess | 82.89 | 0.02288 | 1 | 3623 | 1 | 329.3 | 199.5 | 1.106 | 1.106 | 0.05175 | 0.3333 | 0.3377 | 0.02288 | 0.5257 | 0.999 | 0.9997 |
| alpha=2|I | 0.2503 | 0.00025 | 0.000255 | 1000 | 49.5 | 1 | 0.1358 | 0.6104 | 0.6307 | 0.1875 | 0.1912 | 0.01914 | 0.7936 | 0.2111 | 0.9581 | 1 |
| alpha=2|diag | 65.16 | 0.02285 | 0.02677 | 2852 | 116.2 | 259.2 | 43.3 | 0.2687 | 0.3157 | 0.06574 | 0.07705 | 0.01493 | 0.6179 | 0.08651 | 0.97 | 1 |
| alpha=2|stat | 11.17 | 0.003071 | 0.1086 | 3639 | 8.42 | 41.11 | 21.79 | 1.257 | 1.241 | 0.05153 | 0.3333 | 0.1879 | 0.009403 | 0.5073 | 0.9991 | 0.9997 |
| alpha=2|hess | 41.4 | 0.01981 | 1 | 2089 | 1 | 161.7 | 63.66 | 1.242 | 1.242 | 0.08974 | 0.3333 | 0.1683 | 0.01981 | 0.5088 | 0.9984 | 0.9998 |

## (gamma, Delta) inside the proof (n = 1000, kappa = 1e3, alpha = 1)
Optimal within the proof: c = 2.562, gamma R2 = 0.2192, certified rate/iter = 0.000219 vs textbook 0.000188 (gain x1.169).

| gamma R2 | empirical exponent/pass | certified exponent/pass (Delta optimised) |
|---|---|---|
| 0.125 | 0.5076 | 0.1806 |
| 0.25 | 0.9663 | 0.2156 |
| 0.3333333333333333 | 1.141 | 0.1747 |
| 0.5 | 1.088 | 0 |
| 1.0 | 0.9485 | 0 |
| 2.0 | 0.6591 | 0 |
| 4.0 | 0.02735 | 0 |

## A3: L-Katyusha Lyapunov instrumentation (3 seeds)

rho_lemma = max{c_Z, c_Y, c_W}: one-step factor certified by Lemma 10 of Kovalev, Horvath and Richtarik (ALT 2020; Lemma 4.5 and Theorem 4.6 in the arXiv version 1901.08689 are its Lemma 10 and Theorem 11); the closed-form rate rho of Theorem 11 satisfies rho <= 1 - rho_lemma. Speed-up: mean of log rho_exact over the records with Psi_t >= 1e-10 (all seeds), divided by log rho_lemma.

| problem | theta1 | eta | p | rho_lemma (Lemma 10) | rho exact (geomean) | speed-up exact/lemma | r_var geomean | r_var max | max(rho_exact - rho_lemma) | final D/D0 |
|---|---|---|---|---|---|---|---|---|---|---|
| hard instance (n=16, m=64, κ=10⁴) | 0.03266 | 10.21 | 0.0625 | 0.999 | 0.9952 | 4.668 | 0.125 | 0.5864 | -0.001655 | 0.07303 |
| logistic (n=1000, d=50, κ=10³, α=1) | 0.5 | 0.6667 | 0.001 | 0.9997 | 0.9991 | 2.775 | 0.1463 | 0.2998 | -0.000399 | 5.2e-27 |
| logistic (n=1000, d=50, κ=10⁴, α=1) | 0.2582 | 1.291 | 0.001 | 0.9999 | 0.9995 | 4.117 | 0.1495 | 0.3209 | -0.000276 | 2.57e-14 |
### A3: additional ratios of the A3 note (centering, Bregman co-coercivity, cancellation shares)
| problem | r_cen geomean | r_coco (Bregman) geomean | c(t) median | frac c<0 | frac c>1 | q(t) median | frac q>=1 |
|---|---|---|---|---|---|---|---|
| hard instance (n=16, m=64, κ=10⁴) | 0.9375 | 0.1333 | -1.486 | 0.7716 | 0.1217 | 1.417 | 0.6312 |
| logistic (n=1000, d=50, κ=10³, α=1) | 0.9669 | 0.1513 | -0.6131 | 0.7704 | 0.08143 | 0.6383 | 0.05186 |
| logistic (n=1000, d=50, κ=10⁴, α=1) | 0.9867 | 0.1515 | -0.6195 | 0.6429 | 0.2014 | 1.401 | 0.6505 |

Sign-aware reading of q(t) (A3 note, Table of the cancellation row): q_+(t) = q(t) when <F'(x_t), x_t - w_t> > 0 (c(t) > 0) and 0 otherwise; only then does the cancelled term have the sign a bound without the anchor would have to absorb. Componentwise Lemma 10: max over the records of (E Psi_{t+1} - c_Z Z_t - c_Y Y_t - c_W W_t)/Psi_t (negative = holds). Speed-up vs rho: mean log rho_exact / log(1 - rho), rho = min{sigma/(6 theta1), theta1/(2n)}.

| problem | frac c>0 | frac q_+>=1 | frac q>=1 given c>0 | q median given c>0 | frac c<0 given q>=1 | max componentwise Lemma 10 residual/Psi | speed-up exact/rho |
|---|---|---|---|---|---|---|---|
| hard instance (n=16, m=64, κ=10⁴) | 0.2281 | 0.09861 | 0.4323 | 0.8389 | 0.8438 | -0.000654 | 9.329 |
| logistic (n=1000, d=50, κ=10³, α=1) | 0.2285 | 0.05186 | 0.227 | 0.5553 | 0 | -0.000226 | 3.7 |
| logistic (n=1000, d=50, κ=10⁴, α=1) | 0.3573 | 0.1929 | 0.5398 | 1.092 | 0.7035 | -0.000174 | 8.234 |

## A4b: shared-regularizer (A1-note) and aligned instances (n = 16, eps = 1e-8, 2 seeds)

Slopes: least-squares fits of log(gradients) against log(x) for the per-x geometric means over the seeds (K values of x in the fitting range), with 95% t-intervals with K - 2 degrees of freedom. Gradient counts as in vrlab.algorithms (L-Katyusha: 3 per iteration in expectation, about 3/2 of the stored-reference convention of the A1 note's bounds; slopes are unaffected).

| instance|x-axis|method | slope (x >= 10n) | 95% t-interval | slope in window 10n <= kappa <= kappa_max | 95% t-interval | grads @x 0 | grads @x 1 | grads @x 2 | grads @x 3 | grads @x 4 | grads @x 5 | grads @x 6 | grads @x 7 | grads @x 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared|nominal|SAGA | 0.9452 | [0.897, 0.993] | 0.9937 | [0.990, 0.998] | 2912 | 8513 | 2.79e+04 | 8.32e+04 | 2.76e+05 | 8.26e+05 | 2.68e+06 | 6.9e+06 | 1.55e+07 |
| shared|nominal|L-Katyusha | 0.4895 | [0.467, 0.512] | 0.4997 | [0.479, 0.520] | 2130 | 3640 | 6816 | 1.14e+04 | 2.06e+04 | 3.72e+04 | 7.07e+04 | 1.16e+05 | 1.77e+05 |
| shared|nominal|AGD | 0.5013 | [0.474, 0.529] | 0.5299 | [0.525, 0.535] | 305.9 | 557.2 | 1062 | 1894 | 3557 | 6430 | 1.2e+04 | 1.96e+04 | 2.99e+04 |
| shared|nominal|GD | 0.9435 | [0.896, 0.991] | 0.99 | [0.982, 0.998] | 768.9 | 2169 | 7026 | 2.08e+04 | 6.91e+04 | 2.07e+05 | 6.7e+05 | 1.72e+06 | 3.88e+06 |
| shared|eff|SAGA | 1.031 | [1.014, 1.047] | 1.005 | [0.990, 1.021] | 2912 | 8513 | 2.79e+04 | 8.32e+04 | 2.76e+05 | 8.26e+05 | 2.68e+06 | 6.9e+06 | 1.55e+07 |
| shared|eff|L-Katyusha | 0.5333 | [0.515, 0.552] | 0.5055 | [0.481, 0.530] | 2130 | 3640 | 6816 | 1.14e+04 | 2.06e+04 | 3.72e+04 | 7.07e+04 | 1.16e+05 | 1.77e+05 |
| shared|eff|AGD | 0.5466 | [0.540, 0.554] | 0.536 | [0.527, 0.545] | 305.9 | 557.2 | 1062 | 1894 | 3557 | 6430 | 1.2e+04 | 1.96e+04 | 2.99e+04 |
| shared|eff|GD | 1.029 | [1.011, 1.047] | 1.002 | [0.983, 1.021] | 768.9 | 2169 | 7026 | 2.08e+04 | 6.91e+04 | 2.07e+05 | 6.7e+05 | 1.72e+06 | 3.88e+06 |
| aligned|nominal|SAGA | 0.9986 | [0.998, 0.999] | 0.9986 | [0.998, 0.999] | 2794 | 8313 | 2.76e+04 | 8.26e+04 | 2.75e+05 | 8.24e+05 | 2.75e+06 |
| aligned|nominal|L-Katyusha | 0.5021 | [0.488, 0.517] | 0.5021 | [0.488, 0.517] | 2029 | 3569 | 6684 | 1.12e+04 | 2.02e+04 | 3.54e+04 | 6.78e+04 |
| aligned|nominal|AGD | 0.5114 | [0.509, 0.514] | 0.5114 | [0.509, 0.514] | 1353 | 2406 | 4476 | 7842 | 1.45e+04 | 2.52e+04 | 4.72e+04 |
| aligned|nominal|GD | 0.9991 | [0.999, 0.999] | 0.9991 | [0.999, 0.999] | 1.11e+04 | 3.31e+04 | 1.1e+05 | 3.3e+05 | 1.1e+06 | 3.3e+06 | 1.1e+07 |
| aligned|eff|SAGA | 1.006 | [0.998, 1.014] | 1.006 | [0.998, 1.014] | 2794 | 8313 | 2.76e+04 | 8.26e+04 | 2.75e+05 | 8.24e+05 | 2.75e+06 |
| aligned|eff|L-Katyusha | 0.5058 | [0.488, 0.523] | 0.5058 | [0.488, 0.523] | 2029 | 3569 | 6684 | 1.12e+04 | 2.02e+04 | 3.54e+04 | 6.78e+04 |
| aligned|eff|AGD | 0.5152 | [0.510, 0.521] | 0.5152 | [0.510, 0.521] | 1353 | 2406 | 4476 | 7842 | 1.45e+04 | 2.52e+04 | 4.72e+04 |
| aligned|eff|GD | 1.006 | [0.999, 1.014] | 1.006 | [0.999, 1.014] | 1.11e+04 | 3.31e+04 | 1.1e+05 | 3.3e+05 | 1.1e+06 | 3.3e+06 | 1.1e+07 |
| instance|kappa | kappa_eff = R2/lmin(F'') | kappa_max(m, eps) | m | kappa_F = L_F/mu |
|---|---|---|---|---|
| shared|kappa=100 | 99.97 | 4.94e+04 | 256 | 7.186 |
| shared|kappa=300 | 299.8 | 4.94e+04 | 256 | 19.67 |
| shared|kappa=1000 | 997.7 | 4.94e+04 | 256 | 63.29 |
| shared|kappa=3000 | 2980 | 4.94e+04 | 256 | 187.2 |
| shared|kappa=10000 | 9791 | 4.94e+04 | 256 | 612.8 |
| shared|kappa=30000 | 2.83e+04 | 4.94e+04 | 256 | 1770 |
| shared|kappa=100000 | 8.48e+04 | 4.94e+04 | 256 | 5304 |
| shared|kappa=300000 | 2.05e+05 | 4.94e+04 | 256 | 1.28e+04 |
| shared|kappa=1e+06 | 4.43e+05 | 4.94e+04 | 256 | 2.77e+04 |
| aligned|kappa=100 | 99.99 | 1.98e+05 | 2048 | 99.99 |
| aligned|kappa=300 | 299.9 | 1.98e+05 | 2048 | 299.9 |
| aligned|kappa=1000 | 999.4 | 1.98e+05 | 2048 | 999.4 |
| aligned|kappa=3000 | 2995 | 1.98e+05 | 2048 | 2995 |
| aligned|kappa=10000 | 9944 | 1.98e+05 | 2048 | 9944 |
| aligned|kappa=30000 | 2.95e+04 | 1.98e+05 | 2048 | 2.95e+04 |
| aligned|kappa=100000 | 9.52e+04 | 1.98e+05 | 2048 | 9.52e+04 |

## A2: naive acceleration (one seed)
### hard instance (n=16, m=64, κ=10⁴)
| method | final D/D0 | grads used | stopped | grid best |
|---|---|---|---|---|
| SAGA | 1e-10 | 4.32e+05 | stop_D | None |
| L-SVRG | 1e-10 | 1.94e+06 | stop_D | None |
| L-Katyusha | 9.99e-11 | 3.2e+04 | stop_D | None |
| L-Katyusha (no neg. momentum) | 8.76e+27 | 3406 | diverged | None |
| AGD | 9.65e-11 | 5280 | stop_D | None |
| Nesterov+SVRG | 1e-10 | 9.64e+04 | stop_D | {'beta': 0.95, 'gammaR2': 0.16666666666666666} |
| naive (beta, gamma R2) | grads to 1e-6 | stopped | final D/D0 |
|---|---|---|---|
| beta=0.5,gammaR2=0.167 | 5.56e+05 | stop_D | 1e-10 |
| beta=0.5,gammaR2=0.05 | 1.86e+06 | stop_D | 1e-10 |
| beta=0.5,gammaR2=0.0167 | nan | None | 1.28e-05 |
| beta=0.8,gammaR2=0.167 | 2.22e+05 | stop_D | 1e-10 |
| beta=0.8,gammaR2=0.05 | 7.41e+05 | stop_D | 1e-10 |
| beta=0.8,gammaR2=0.0167 | 2.23e+06 | stop_D | 1e-10 |
| beta=0.9,gammaR2=0.167 | 1.11e+05 | stop_D | 1e-10 |
| beta=0.9,gammaR2=0.05 | 3.71e+05 | stop_D | 1e-10 |
| beta=0.9,gammaR2=0.0167 | 1.12e+06 | stop_D | 1e-10 |
| beta=0.95,gammaR2=0.167 | 5.51e+04 | stop_D | 1e-10 |
| beta=0.95,gammaR2=0.05 | 1.84e+05 | stop_D | 1e-10 |
| beta=0.95,gammaR2=0.0167 | 5.56e+05 | stop_D | 1e-10 |
| beta=kappa,gammaR2=0.167 | nan | diverged | 9.87e+27 |
| beta=kappa,gammaR2=0.05 | 7.26e+04 | stop_D | 1e-10 |
| beta=kappa,gammaR2=0.0167 | 2.19e+05 | stop_D | 1e-10 |
### logistic (n=1000, d=50, κ=10⁴, α=1)
| method | final D/D0 | grads used | stopped | grid best |
|---|---|---|---|---|
| SAGA | 9.83e-11 | 8.78e+04 | stop_D | None |
| L-SVRG | 9.88e-11 | 3.98e+05 | stop_D | None |
| L-Katyusha | 1.6e-11 | 1.06e+05 | stop_D | None |
| L-Katyusha (no neg. momentum) | 9.94e-11 | 4.9e+04 | stop_D | None |
| AGD | 6.1e-08 | 2e+05 | None | None |
| Nesterov+SVRG | 8.68e-11 | 4.66e+04 | stop_D | {'beta': 0.9, 'gammaR2': 0.16666666666666666} |
| naive (beta, gamma R2) | grads to 1e-6 | stopped | final D/D0 |
|---|---|---|---|
| beta=0.5,gammaR2=0.167 | 1.15e+05 | stop_D | 9.79e-11 |
| beta=0.5,gammaR2=0.05 | 3.8e+05 | None | 5.3e-10 |
| beta=0.5,gammaR2=0.0167 | nan | None | 0.000309 |
| beta=0.8,gammaR2=0.167 | 4.38e+04 | stop_D | 9.79e-11 |
| beta=0.8,gammaR2=0.05 | 1.53e+05 | stop_D | 9.92e-11 |
| beta=0.8,gammaR2=0.0167 | 4.5e+05 | None | 1.45e-08 |
| beta=0.9,gammaR2=0.167 | 2.78e+04 | stop_D | 8.68e-11 |
| beta=0.9,gammaR2=0.05 | 7.77e+04 | stop_D | 9.62e-11 |
| beta=0.9,gammaR2=0.0167 | 2.33e+05 | stop_D | 9.9e-11 |
| beta=0.95,gammaR2=0.167 | 5.32e+04 | stop_D | 6.17e-11 |
| beta=0.95,gammaR2=0.05 | 3.29e+04 | stop_D | 9.07e-11 |
| beta=0.95,gammaR2=0.0167 | 1.15e+05 | stop_D | 9.86e-11 |
| beta=kappa,gammaR2=0.167 | nan | None | 1.264 |
| beta=kappa,gammaR2=0.05 | 3.93e+04 | stop_D | 9.04e-11 |
| beta=kappa,gammaR2=0.0167 | 4.23e+04 | stop_D | 9.76e-11 |

## A4: gradient complexity vs kappa (n = 16, m = 256, eps = 1e-8, 3 seeds)

Slopes: least-squares fits of log(gradients) against log(kappa) for the per-kappa geometric means over the seeds, over kappa >= 10n (K values of kappa), with 95% t-intervals with K - 2 degrees of freedom. Gradient counts as in vrlab.algorithms: L-SVRG, L-Katyusha and Nesterov+SVRG are charged 3 per iteration in expectation, about 3/2 of the stored-reference convention of the A1 note's bounds (slopes are unaffected). Nesterov+SVRG: best of 6 (beta, gamma) settings, selected on seed 0 on (H) and re-run and re-selected on every seed on (R), whose instance depends on the seed.

| instance|method | slope (kappa >= 10n) | 95% t-interval (K - 2 d.o.f.) | grads @kappa 0 | grads @kappa 1 | grads @kappa 2 | grads @kappa 3 | grads @kappa 4 | grads @kappa 5 | grads @kappa 6 |
|---|---|---|---|---|---|---|---|---|---|
| hard|SAGA | 1.022 | [0.997, 1.047] | 2927 | 8526 | 2.8e+04 | 8.38e+04 | 2.83e+05 | 8.88e+05 | 3.27e+06 |
| hard|L-SVRG | 1.021 | [0.996, 1.047] | 1.35e+04 | 3.85e+04 | 1.26e+05 | 3.77e+05 | 1.27e+06 | 4e+06 | 1.47e+07 |
| hard|L-Katyusha | 0.5246 | [0.498, 0.551] | 2178 | 3654 | 6748 | 1.14e+04 | 2.11e+04 | 3.9e+04 | 7.85e+04 |
| hard|L-Katyusha (no neg. momentum) | nan | n/a | 2226 | 3615 | nan | nan | nan | nan | nan |
| hard|Nesterov+SVRG | 0.8065 | [0.618, 0.995] | 1734 | 4247 | 5982 | 1.33e+04 | 6.34e+04 | 1.51e+05 | 3.06e+05 |
| hard|AGD | 0.5496 | [0.539, 0.560] | 279.9 | 542.1 | 1055 | 1897 | 3604 | 6740 | 1.34e+04 |
| hard|GD | 1.026 | [1.005, 1.048] | 663.1 | 2066 | 6938 | 2.09e+04 | 7.06e+04 | 2.22e+05 | 8.18e+05 |
| random|SAGA | 1.003 | [1.002, 1.003] | 3259 | 9777 | 3.27e+04 | 9.83e+04 | 3.29e+05 | 9.89e+05 | 3.31e+06 |
| random|L-SVRG | 1.002 | [1.001, 1.003] | 1.5e+04 | 4.41e+04 | 1.47e+05 | 4.42e+05 | 1.48e+06 | 4.45e+06 | 1.49e+07 |
| random|L-Katyusha | 0.5002 | [0.496, 0.505] | 2476 | 4315 | 7833 | 1.34e+04 | 2.47e+04 | 4.31e+04 | 7.85e+04 |
| random|L-Katyusha (no neg. momentum) | nan | n/a | 2483 | 4138 | nan | nan | nan | nan | nan |
| random|Nesterov+SVRG | 0.8144 | [0.649, 0.980] | 1933 | 3962 | 6816 | 1.56e+04 | 7.39e+04 | 1.68e+05 | 3.09e+05 |
| random|AGD | 0.5258 | [0.513, 0.539] | 312.9 | 618.3 | 1214 | 2183 | 4084 | 7172 | 1.32e+04 |
| random|GD | 1.007 | [1.003, 1.011] | 742 | 2372 | 8098 | 2.45e+04 | 8.21e+04 | 2.47e+05 | 8.27e+05 |

## B4: sampling order and the memory term of SAGA

Exponents are contraction exponents per pass (n iterations) of a squared error, in nats; c = gamma n lambda. Every quantity below is defined in the entry 'definitions' of results/B4_summary.json.

### B4 (M): identical-Hessian model, i.i.d. indices (typical run: median [min, max] over 9 seeds, fit over the passes 30-150)
| c | E e^2: limit min(2c, 1) | E e^2 exact, n=100 | E e^2 exact, n=1000 | E e^2 exact, n=10000 | typical run, n=100 | typical run, n=1000 | typical run, n=10000 |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.5 | 0.5005 | 0.5 | 0.5 | 0.4981 [0.4955, 0.504] | 0.5006 [0.4948, 0.5073] | 0.4996 [0.4938, 0.5033] |
| 0.5 | 1 | 0.9574 | 0.985 | 0.9951 | 1.006 [0.982, 1.025] | 1.006 [0.986, 1.056] | 1 [0.9877, 1.009] |
| 1 | 1 | 0.995 | 0.9995 | 1 | 1.519 [1.464, 1.615] | 1.611 [1.525, 1.642] | 1.684 [1.635, 1.713] |
| 2 | 1 | 0.9915 | 0.9992 | 0.9999 | 1.488 [1.457, 1.543] | 1.579 [1.504, 1.68] | 1.709 [1.627, 1.773] |
| 4 | 1 | 0.9815 | 0.9982 | 0.9998 | 1.407 [1.343, 1.49] | 1.516 [1.433, 1.569] | 1.636 [1.568, 1.742] |
| 8 | 1 | 0.9602 | 0.9962 | 0.9996 | 1.266 [1.218, 1.375] | 1.473 [1.444, 1.54] | 1.595 [1.494, 1.633] |
| 16 | 1 | 0.9145 | 0.9922 | 0.9992 | 1.13 [1.065, 1.174] | 1.424 [1.343, 1.469] | 1.524 [1.471, 1.59] |
| 32 | 1 | 0.8099 | 0.984 | 0.9984 | 0.9182 [0.8735, 0.9599] | 1.344 [1.285, 1.404] | 1.49 [1.443, 1.565] |
### B4 (M): random reshuffling
| c | fluid law (R3) | squared mean (first moments), n=100 | squared mean (first moments), n=1000 | squared mean (first moments), n=10000 | E e^2 exact (second moments), n=100 | E e^2 exact (second moments), n=1000 | E e^2 exact (second moments), n=10000 | simulated run, n=100 | simulated run, n=1000 | simulated run, n=10000 | reshuffled / i.i.d. typical, n=100 | reshuffled / i.i.d. typical, n=1000 | reshuffled / i.i.d. typical, n=10000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0.4445 | 0.4455 | 0.4446 | 0.4446 | 0.4455 | 0.4446 | 0.4446 | 0.4456 [0.4455, 0.4457] | 0.4447 [0.4446, 0.4447] | 0.4446 [0.4445, 0.4446] | 0.8945 | 0.8882 | 0.8898 |
| 0.5 | 0.7772 | 0.7801 | 0.7775 | 0.7772 | 0.7801 | 0.7775 | 0.7772 | 0.7802 [0.78, 0.7805] | 0.7775 [0.7774, 0.7776] | 0.7772 [0.7772, 0.7772] | 0.7759 | 0.7729 | 0.7772 |
| 1 | 1.187 | 1.194 | 1.188 | 1.187 | 1.194 | 1.188 | 1.187 | 1.194 [1.193, 1.194] | 1.188 [1.188, 1.188] | 1.187 [1.187, 1.187] | 0.7861 | 0.7375 | 0.705 |
| 2 | 1.537 | 1.546 | 1.538 | 1.537 | 1.546 | 1.538 | 1.537 | 1.546 [1.546, 1.547] | 1.538 [1.538, 1.538] | 1.537 [1.537, 1.537] | 1.039 | 0.9741 | 0.8992 |
| 4 | 1.752 | 1.763 | 1.753 | 1.752 | 1.763 | 1.753 | 1.752 | 1.763 [1.761, 1.764] | 1.753 [1.753, 1.754] | 1.752 [1.752, 1.753] | 1.253 | 1.156 | 1.071 |
| 8 | 1.863 | 1.873 | 1.864 | 1.863 | 1.873 | 1.864 | 1.863 | 1.873 [1.87, 1.874] | 1.864 [1.863, 1.864] | 1.863 [1.862, 1.863] | 1.48 | 1.265 | 1.168 |
| 16 | 1.923 | 1.933 | 1.924 | 1.923 | 1.933 | 1.924 | 1.923 | 1.931 [1.929, 1.936] | 1.924 [1.924, 1.925] | 1.923 [1.923, 1.923] | 1.709 | 1.351 | 1.262 |
| 32 | 1.959 | 1.969 | 1.96 | 1.959 | 1.603 | 1.96 | 1.959 | 1.619 [1.584, 1.678] | 1.96 [1.959, 1.961] | 1.959 [1.958, 1.959] | 1.763 | 1.458 | 1.315 |
| n | max over c of |simulated reshuffled median - fluid law| | max over c of |simulated reshuffled median - exact E e^2 exponent| | largest seed spread, reshuffled | largest seed spread, i.i.d. |
|---|---|---|---|---|
| n=100 | 0.3394 | 0.01619 | 0.09372 | 0.1579 |
| n=1000 | 0.001293 | 0.000308 | 0.001275 | 0.1759 |
| n=10000 | 0.000149 | 6.91e-05 | 0.000819 | 0.174 |
### B4 (M): fixed order (cyclic; any fixed permutation is equivalent in the model)

Critical step c_* = 17.702277 (omega_* = 5.950173); best exponent of a fixed order 1.52636 at c = 2.14782; stability threshold of the companion matrix: 14.784 (n=100), 17.357 (n=1000).

| c | limit (R2) | companion matrix, n=100 | companion matrix, n=1000 | simulated run, n=100 | simulated run, n=1000 | simulated run, n=10000 |
|---|---|---|---|---|---|---|
| 0.25 | 0.4368 | 0.4377 | 0.4369 | 0.4377 | 0.4369 | 0.4368 |
| 0.5 | 0.7559 | 0.7586 | 0.7562 | 0.7586 | 0.7562 | 0.7559 |
| 1 | 1.149 | 1.155 | 1.15 | 1.155 | 1.15 | 1.149 |
| 2 | 1.497 | 1.505 | 1.498 | 1.505 | 1.498 | 1.497 |
| 4 | 0.7017 | 0.6698 | 0.6985 | 0.6698 | 0.6984 | 0.7012 |
| 8 | 0.1937 | 0.1648 | 0.1908 | 0.1648 | 0.1908 | 0.1934 |
| 16 | 0.01075 | -0.009338 | 0.008754 | -0.009318 | 0.008758 | 0.01055 |
| 32 | -0.02507 | -0.03663 | -0.02622 | -0.0359 | -0.02607 | -0.02506 |
### B4 (M): expectation against single runs, i.i.d. indices, n = 1000, c = 2 (9 seeds)

Exact exponent of E e^2: 0.9992 per pass; n times the amplitude of its slow mode: 1.774 (limit 2c^3/(2c-1)^2 = 1.778); pass at which the last slot of the initial memory is refreshed: 7.723 [5.722, 9.116].

| pass | exact E e^2 | mean of the simulated runs | median run | largest run |
|---|---|---|---|---|
| 10 | 8.12e-08 | 8.67e-08 | 1.15e-08 | 6.23e-07 |
| 20 | 3.72e-12 | 8.02e-16 | 7.34e-17 | 4.88e-15 |
| 30 | 1.7e-16 | 1.73e-22 | 4.65e-23 | 7.52e-22 |
| 40 | 7.79e-21 | 1.23e-30 | 8.02e-31 | 4.59e-30 |
| 50 | 3.57e-25 | 7.48e-37 | 5.64e-38 | 3.08e-36 |
| 60 | 1.63e-29 | 5.92e-44 | 5.2e-46 | 2.98e-43 |
### B4 (L): logistic regression (n = 1000, d = 50, theta_0 = 0, 5 seeds, 300 passes at most; one data draw per problem)

Runs by status (cyclic: one run per step, the order does not use the seed): iid 449 converged, 91 stalled (0 growing, 91 slow), 0 diverged (0 exploded, 0 ended above the initial error); rr 445 converged, 95 stalled (0 growing, 95 slow), 0 diverged (0 exploded, 0 ended above the initial error); so 230 converged, 219 stalled (92 growing, 127 slow), 91 diverged (0 exploded, 91 ended above the initial error); cyclic 46 converged, 43 stalled (17 growing, 26 slow), 19 diverged (0 exploded, 19 ended above the initial error). Largest relative difference between the batched loop of the script and vrlab.algorithms.saga: 1.12e-06.

| problem | kappa/n | n lmin(H*)/R2 | n lmax(H*)/R2 | i.i.d. textbook step: exponent | passes to 1e-10 | i.i.d. best step: exponent | reshuffled best step: exponent | shuffled once best step: exponent | cyclic best step: exponent | reshuffled / i.i.d. (best exponents) | fewest passes to 1e-10 (first passage), i.i.d. | reshuffled | saving | fewest passes from which the error stays below 1e-10, i.i.d. | reshuffled | saving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kappa=10, alpha=0 | 0.01 | 106.5 | 115.7 | 1.27 [0.716, 1.3] | 14.3 | 0.01562: 1.61 [1.4, 1.72] | 0.5: 1.99 [1.9, 2.01] | 0.01562: 1.43 [1.43, 1.43] | 0.01562: 1.43 [1.43, 1.43] | 1.234 | 12.65 | 10.09 | 0.2025 | 12.65 | 12.01 | 0.05072 |
| kappa=10, alpha=1 | 0.01 | 101 | 158.6 | 1.26 [0.781, 1.29] | 14.08 | 0.01562: 1.61 [1.31, 1.82] | 1: 2.01 [1.87, 2.05] | 0.01562: 1.42 [1.42, 1.42] | 0.01562: 1.42 [1.42, 1.42] | 1.25 | 12.44 | 10.08 | 0.19 | 12.54 | 12.01 | 0.04231 |
| kappa=10, alpha=2 | 0.01 | 100 | 169.9 | 1.3 [0.79, 1.39] | 13.87 | 0.007812: 1.5 [1.44, 1.76] | 0.5: 1.99 [1.9, 2.01] | 0.01562: 1.28 [1.27, 1.3] | 0.01562: 1.28 [1.28, 1.28] | 1.325 | 11.63 | 10.09 | 0.1328 | 13.42 | 12 | 0.1054 |
| kappa=100, alpha=0 | 0.1 | 16.09 | 24.71 | 1.13 [1.09, 1.3] | 18.19 | 0.0625: 1.34 [1.19, 1.51] | 0.5: 1.89 [1.87, 1.89] | 0.0625: 1.18 [1.18, 1.18] | 0.0625: 1.18 [1.18, 1.18] | 1.411 | 16.33 | 12.09 | 0.2592 | 16.33 | 12.11 | 0.2585 |
| kappa=100, alpha=1 | 0.1 | 11 | 68.81 | 1.26 [1.16, 1.32] | 15.71 | 0.0625: 1.39 [1.24, 1.42] | 1: 1.88 [1.86, 1.92] | 0.0625: 0.632 [0.618, 0.959] | 0.0625: 0.623 [0.623, 0.623] | 1.353 | 15.14 | 11.27 | 0.2557 | 15.14 | 12.06 | 0.2032 |
| kappa=100, alpha=2 | 0.1 | 10.03 | 73.05 | 1.3 [1.19, 1.51] | 14.3 | 0.125: 1.39 [1.29, 1.6] | 1: 1.93 [1.89, 1.96] | 0.0625: 0.56 [0.556, 0.574] | 0.03125: 0.549 [0.549, 0.549] | 1.383 | 13.88 | 11.11 | 0.1994 | 13.88 | 12.04 | 0.1325 |
| kappa=1000, alpha=0 | 1 | 3.275 | 11.19 | 1.06 [1.03, 1.14] | 19.67 | 0.25: 1.06 [1.03, 1.14] | 0.5: 1.44 [1.44, 1.45] | 0.25: 1.02 [1.02, 1.02] | 0.25: 1.02 [1.02, 1.02] | 1.362 | 19.67 | 16.16 | 0.1784 | 19.67 | 16.16 | 0.1784 |
| kappa=1000, alpha=1 | 1 | 1.737 | 43.71 | 0.911 [0.9, 0.932] | 22.25 | 0.5: 1.06 [1.02, 1.09] | 1: 1.54 [1.53, 1.55] | 0.125: 0.411 [0.411, 0.411] | 0.125: 0.411 [0.411, 0.411] | 1.456 | 18.89 | 14.5 | 0.2325 | 19.38 | 15.03 | 0.2245 |
| kappa=1000, alpha=2 | 1 | 1.02 | 50.47 | 0.532 [0.531, 0.534] | 37.13 | 1: 1.07 [0.996, 1.15] | 2: 1.49 [1.48, 1.57] | 0.125: 0.249 [0.249, 0.249] | 0.125: 0.249 [0.249, 0.249] | 1.393 | 19.18 | 14.16 | 0.2616 | 19.31 | 14.22 | 0.2635 |
| kappa=10000, alpha=0 | 10 | 0.9863 | 8.006 | 0.494 [0.493, 0.497] | 42.71 | 0.5: 0.931 [0.903, 0.938] | 1: 1.01 [0.951, 1.04] | 0.5: 0.494 [0.48, 0.502] | 0.5: 0.477 [0.477, 0.477] | 1.082 | 23.14 | 21.27 | 0.08107 | 23.14 | 21.27 | 0.08107 |
| kappa=10000, alpha=1 | 10 | 0.497 | 29.98 | 0.248 [0.248, 0.249] | 86.69 | 1: 0.895 [0.837, 0.905] | 2: 0.841 [0.81, 0.852] | 0.25: 0.189 [0.175, 0.229] | 0.25: 0.183 [0.183, 0.183] | 0.9406 | 24.12 | 25.07 | -0.03952 | 24.12 | 25.07 | -0.03952 |
| kappa=10000, alpha=2 | 10 | 0.1142 | 37.55 | 0.0647 [0.0647, 0.0647] | n/a | 2: 0.481 [0.473, 0.493] | 2: 0.433 [0.432, 0.434] | n/a | n/a | 0.9005 | 40.56 | 45.34 | -0.1179 | 40.56 | 45.34 | -0.1179 |
### B4 (L): the law (R3) at c = gamma n lmin(H*) against the reshuffled runs (measured / law, by gamma R2)
| problem | gamma R2 = 0.007812 | gamma R2 = 0.01562 | gamma R2 = 0.03125 | gamma R2 = 0.0625 | gamma R2 = 0.125 | gamma R2 = 0.25 | gamma R2 = 0.5 | gamma R2 = 1 | gamma R2 = 2 |
|---|---|---|---|---|---|---|---|---|---|
| kappa=10, alpha=0 | 1.102 / 1.078 (+2.2%) | 1.477 / 1.457 (+1.4%) | 1.719 / 1.708 (+0.7%) | 1.843 / 1.840 (+0.1%) | 1.924 / 1.910 (+0.7%) | 1.954 / 1.951 (+0.2%) | 1.987 / 1.974 (+0.6%) | 1.986 / 1.987 (-0.0%) | 1.637 / 1.993 (-17.9%) |
| kappa=10, alpha=1 | 1.076 / 1.047 (+2.8%) | 1.462 / 1.432 (+2.1%) | 1.714 / 1.693 (+1.2%) | 1.836 / 1.833 (+0.2%) | 1.925 / 1.906 (+1.0%) | 1.956 / 1.949 (+0.4%) | 1.987 / 1.973 (+0.7%) | 2.014 / 1.986 (+1.4%) | 1.316 / 1.993 (-34.0%) |
| kappa=10, alpha=2 | 1.075 / 1.041 (+3.3%) | 1.471 / 1.427 (+3.1%) | 1.735 / 1.691 (+2.6%) | 1.852 / 1.832 (+1.1%) | 1.932 / 1.905 (+1.4%) | 1.961 / 1.948 (+0.6%) | 1.993 / 1.973 (+1.0%) | 1.907 / 1.986 (-4.0%) | 1.215 / 1.993 (-39.0%) |
| kappa=100, alpha=0 | 0.245 / 0.238 (+3.1%) | 0.460 / 0.447 (+3.0%) | 0.802 / 0.780 (+2.7%) | 1.218 / 1.191 (+2.3%) | 1.565 / 1.539 (+1.7%) | 1.770 / 1.754 (+1.0%) | 1.885 / 1.863 (+1.2%) | 1.695 / 1.923 (-11.9%) | 0.908 / 1.959 (-53.6%) |
| kappa=100, alpha=1 | 0.176 / 0.166 (+6.4%) | 0.338 / 0.318 (+6.3%) | 0.617 / 0.582 (+6.0%) | 1.014 / 0.964 (+5.2%) | 1.420 / 1.363 (+4.2%) | 1.710 / 1.652 (+3.5%) | 1.852 / 1.812 (+2.2%) | 1.880 / 1.894 (-0.8%) | 1.175 / 1.942 (-39.5%) |
| kappa=100, alpha=2 | 0.157 / 0.151 (+3.8%) | 0.303 / 0.292 (+3.8%) | 0.560 / 0.539 (+3.9%) | 0.949 / 0.908 (+4.5%) | 1.387 / 1.314 (+5.6%) | 1.704 / 1.622 (+5.1%) | 1.851 / 1.797 (+3.0%) | 1.926 / 1.886 (+2.1%) | 1.585 / 1.937 (-18.2%) |
| kappa=1000, alpha=0 | n/a | 0.100 / 0.100 (+0.0%) | 0.196 / 0.196 (-0.0%) | 0.372 / 0.372 (-0.1%) | 0.668 / 0.669 (-0.2%) | 1.065 / 1.069 (-0.4%) | 1.445 / 1.449 (-0.3%) | 1.176 / 1.703 (-31.0%) | 0.303 / 1.838 (-83.5%) |
| kappa=1000, alpha=1 | n/a | n/a | 0.115 / 0.106 (+8.6%) | 0.220 / 0.207 (+6.1%) | 0.417 / 0.393 (+6.3%) | 0.745 / 0.700 (+6.4%) | 1.171 / 1.104 (+6.0%) | 1.542 / 1.476 (+4.5%) | 0.989 / 1.719 (-42.4%) |
| kappa=1000, alpha=2 | n/a | n/a | n/a | 0.130 / 0.124 (+4.7%) | 0.251 / 0.241 (+4.3%) | 0.471 / 0.452 (+4.2%) | 0.821 / 0.788 (+4.1%) | 1.250 / 1.199 (+4.3%) | 1.493 / 1.545 (-3.3%) |
| kappa=10000, alpha=0 | n/a | n/a | n/a | 0.120 / 0.120 (-0.0%) | 0.233 / 0.233 (-0.1%) | 0.438 / 0.439 (-0.3%) | 0.764 / 0.769 (-0.7%) | 1.006 / 1.179 (-14.7%) | 0.131 / 1.531 (-91.4%) |
| kappa=10000, alpha=1 | n/a | n/a | n/a | n/a | 0.121 / 0.121 (-0.0%) | 0.235 / 0.235 (-0.1%) | 0.441 / 0.442 (-0.2%) | 0.770 / 0.774 (-0.5%) | 0.841 / 1.184 (-28.9%) |
| kappa=10000, alpha=2 | n/a | n/a | n/a | n/a | n/a | n/a | 0.119 / 0.111 (+7.1%) | 0.229 / 0.217 (+5.4%) | 0.433 / 0.411 (+5.4%) |

gamma R2 <= 1/2: 65 (problem, step) pairs, largest |relative error| 8.6% (kappa=1000, alpha=1, gamma R2 = 0.03125), median relative error +2.2%, 65 pairs within 10%.

gamma R2 >= 1: 24 (problem, step) pairs, largest |relative error| 91.4% (kappa=10000, alpha=0, gamma R2 = 2), median relative error -13.3%, 11 pairs within 10%.

### B4 (L): fixed orders against c_max = gamma n lmax(H*) (shuffled-once median / cyclic / law (R2); * = failure)

Law (R2): min of the limiting fixed-order exponent of the model at c_min and at c_max. Model threshold c_* = 17.70 (companion matrix at n = 1000: 17.36).

| problem | gamma R2 = 0.007812 | gamma R2 = 0.01562 | gamma R2 = 0.03125 | gamma R2 = 0.0625 | gamma R2 = 0.125 | gamma R2 = 0.25 | gamma R2 = 0.5 | gamma R2 = 1 | gamma R2 = 2 |
|---|---|---|---|---|---|---|---|---|---|
| kappa=10, alpha=0 | c_max 0.904: 1.067 / 1.067 / 1.044 | c_max 1.81: 1.435 / 1.435 / 1.415 | c_max 3.62: 0.8265 / 0.8281 / 0.8112 | c_max 7.23: 0.2439 / 0.2436 / 0.2431 | c_max 14.5: 0.0242* / 0.02413* / 0.02441 | c_max 28.9: n/a* / n/a* / -0.02384 | c_max 57.9: n/a* / n/a* / -0.02335 | c_max 116: n/a* / n/a* / -0.01495 | c_max 231: -0.00798* / -0.008344* / -0.00831 |
| kappa=10, alpha=1 | c_max 1.24: 1.041 / 1.041 / 1.014 | c_max 2.48: 1.419 / 1.419 / 1.306 | c_max 4.96: 0.5022 / 0.4994 / 0.5002 | c_max 9.91: 0.107 / 0.1064 / 0.1111 | c_max 19.8: -0.003254* / -0.006501* / -0.009099 | c_max 39.6: n/a* / n/a* / -0.02553 | c_max 79.3: n/a* / n/a* / -0.02389 | c_max 159: n/a* / n/a* / -0.01557 | c_max 317: -0.006203* / -0.006324* / -0.008715 |
| kappa=10, alpha=2 | c_max 1.33: 1.041 / 1.041 / 1.008 | c_max 2.65: 1.275 / 1.276 / 1.206 | c_max 5.31: 0.4516 / 0.4366 / 0.4443 | c_max 10.6: 0.0857 / 0.08498 / 0.09026 | c_max 21.2: -0.0122* / -0.01168* / -0.01337 | c_max 42.5: n/a* / n/a* / -0.02523 | c_max 84.9: n/a* / n/a* / -0.02398 | c_max 170: n/a* / n/a* / -0.01568 | c_max 340: -0.01199* / -0.009983* / -0.008789 |
| kappa=100, alpha=0 | c_max 0.193: 0.2426 / 0.2427 / 0.2355 | c_max 0.386: 0.4519 / 0.4519 / 0.439 | c_max 0.772: 0.779 / 0.7791 / 0.7591 | c_max 1.54: 1.176 / 1.176 / 1.153 | c_max 3.09: 0.981 / 0.9852 / 1.002 | c_max 6.18: 0.2946 / 0.2957 / 0.336 | c_max 12.4: 0.0231* / 0.02371* / 0.05293 | c_max 24.7: n/a* / n/a* / -0.02001 | c_max 49.4: n/a* / n/a* / -0.02512 |
| kappa=100, alpha=1 | c_max 0.538: 0.1749 / 0.1749 / 0.1644 | c_max 1.08: 0.3334 / 0.3334 / 0.3138 | c_max 2.15: 0.6025 / 0.6026 / 0.5691 | c_max 4.3: 0.6324 / 0.6231 / 0.6287 | c_max 8.6: 0.1479 / 0.1466 / 0.1624 | c_max 17.2: -0.00711* / -0.006803* / 0.002767 | c_max 34.4: -0.02101* / n/a* / -0.02551 | c_max 68.8: n/a* / n/a* / -0.02034 | c_max 138: n/a* / n/a* / -0.01524 |
| kappa=100, alpha=2 | c_max 0.571: 0.1561 / 0.1561 / 0.1504 | c_max 1.14: 0.2994 / 0.2994 / 0.2884 | c_max 2.28: 0.5487 / 0.5488 / 0.5278 | c_max 4.57: 0.5602 / 0.5487 / 0.5723 | c_max 9.13: 0.1312 / 0.1301 / 0.1393 | c_max 18.3: -0.000824* / -0.00092* / -0.002793 | c_max 36.5: 0.000984* / 0.001117* / -0.02564 | c_max 73.1: -0.003038* / n/a* / -0.01958 | c_max 146: n/a* / n/a* / -0.01155 |
| kappa=1000, alpha=0 | c_max 0.0874: 0.0506 / 0.0506 / 0.05051 | c_max 0.175: 0.0997 / 0.0997 / 0.0997 | c_max 0.35: 0.194 / 0.194 / 0.1941 | c_max 0.699: 0.3665 / 0.3665 / 0.3669 | c_max 1.4: 0.6499 / 0.6499 / 0.6523 | c_max 2.8: 1.024 / 1.023 / 1.035 | c_max 5.59: 0.2765 / 0.2687 / 0.4047 | c_max 11.2: -0.005452* / -0.004378* / 0.07613 | c_max 22.4: n/a* / n/a* / -0.01605 |
| kappa=1000, alpha=1 | c_max 0.341: 0.03193 / 0.03193 / 0.02695 | c_max 0.683: 0.061 / 0.061 / 0.05353 | c_max 1.37: 0.1148 / 0.1148 / 0.1056 | c_max 2.73: 0.2179 / 0.218 / 0.2052 | c_max 5.46: 0.4108 / 0.4112 / 0.3864 | c_max 10.9: 0.0567 / 0.05801 / 0.08235 | c_max 21.9: -0.02438* / -0.009612* / -0.0149 | c_max 43.7: -0.008647* / -0.003235* / -0.02506 | c_max 87.4: n/a* / n/a* / -0.01733 |
| kappa=1000, alpha=2 | c_max 0.394: 0.02 / 0.02 / 0.01587 | c_max 0.789: 0.03632 / 0.03632 / 0.03161 | c_max 1.58: 0.06779 / 0.06779 / 0.0627 | c_max 3.15: 0.1292 / 0.1292 / 0.1233 | c_max 6.31: 0.2486 / 0.2487 / 0.2384 | c_max 12.6: 0.03695* / 0.03734* / 0.04855 | c_max 25.2: -0.002007* / -0.001878* / -0.02068 | c_max 50.5: -0.000477* / 0.000599* / -0.02389 | c_max 101: 0.001737* / 0.001012* / -0.01558 |
| kappa=10000, alpha=0 | c_max 0.0625: 0.01669 / 0.01669 / 0.01535 | c_max 0.125: 0.03105 / 0.03105 / 0.03058 | c_max 0.25: 0.06073 / 0.06073 / 0.06069 | c_max 0.5: 0.1194 / 0.1194 / 0.1195 | c_max 1: 0.2308 / 0.2308 / 0.2312 | c_max 2: 0.4293 / 0.4293 / 0.4316 | c_max 4: 0.4943 / 0.4766 / 0.7009 | c_max 8.01: 0.0181* / 0.01448* / 0.1933 | c_max 16: -0.000789* / -0.000805* / 0.01066 |
| kappa=10000, alpha=1 | c_max 0.234: 0.00945 / 0.00945 / 0.007751 | c_max 0.469: 0.01664 / 0.01664 / 0.01547 | c_max 0.937: 0.03124 / 0.03124 / 0.03082 | c_max 1.87: 0.0612 / 0.0612 / 0.06116 | c_max 3.75: 0.1203 / 0.1203 / 0.1204 | c_max 7.5: 0.1894 / 0.1827 / 0.2247 | c_max 15: -0.01242* / -0.009574* / 0.01919 | c_max 30: -0.001506* / -6.31e-05* / -0.02437 | c_max 60: n/a* / n/a* / -0.02202 |
| kappa=10000, alpha=2 | c_max 0.293: 0.003726 / 0.003726 / 0.001784 | c_max 0.587: 0.006115 / 0.006115 / 0.003567 | c_max 1.17: 0.01036 / 0.01036 / 0.007127 | c_max 2.35: 0.01849 / 0.01849 / 0.01423 | c_max 4.69: 0.03431 / 0.03432 / 0.02835 | c_max 9.39: 0.06385 / 0.06389 / 0.0563 | c_max 18.8: -0.001255* / -0.000875* / -0.005076 | c_max 37.5: -0.000848* / -0.00073* / -0.02563 | c_max 75.1: -0.008444* / -0.01053* / -0.01924 |

so: 43 failing pairs out of 108; largest c_max without failure 10.93, smallest c_max with a failure 8.006; 8 failures below c_*, 0 pairs above c_* without failure. Slow or failing (wider notion): 55 pairs, smallest c_max 4.301, largest c_max of a pair that is not slow 9.387, 20 below c_*.

cyclic: 43 failing pairs out of 108; largest c_max without failure 10.93, smallest c_max with a failure 8.006; 8 failures below c_*, 0 pairs above c_* without failure. Slow or failing (wider notion): 55 pairs, smallest c_max 4.301, largest c_max of a pair that is not slow 9.387, 20 below c_*.

Law (R2) against the measured exponents (108 pairs): largest |difference| 0.2066 (shuffled once) and 0.2243 (cyclic) nats per pass; median |difference| 0.008099 and 0.007692.

### B4 (S): instance (S) of the A1 note (n = 256, m = 12, theta_0 = 0, 32 seeds): gradient counts / n at which the seed mean of the relative squared error falls below eps
| instance | eps | smallest m allowed | bound, i.i.d. span methods | bound, any sequence | i.i.d. textbook step | i.i.d. best (gamma R2) | typical run | i.i.d. best / i.i.d. bound | reshuffled textbook step | reshuffled best (gamma R2) | typical run | reshuffled best / i.i.d. bound | reshuffled best / any-sequence bound | reshuffled best / i.i.d. best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kappa/n=0.1 | 1e-08 | 3 | 11.14 | 2.35 | 13.33 | 13.33 (0.25) | 11.63 | 1.20 | 11.42 | 10.06 (2) | 9.15 | 0.90 | 4.28 | 0.75 |
| kappa/n=0.1 | 1e-16 | 5 | 29.50 | 4.79 | 27.02 | 26.40 (0.125) | 23.68 | 0.89 | 23.11 | 19.08 (2) | 19.07 | 0.65 | 3.99 | 0.72 |
| kappa/n=0.1 | 1e-24 | 8 | 47.86 | 7.23 | 40.46 | 39.47 (0.125) | 35.58 | 0.82 | 34.19 | 29.05 (2) | 28.20 | 0.61 | 4.02 | 0.74 |
| kappa/n=0.3333 | 1e-08 | 4 | 13.45 | 3.35 | 14.95 | 14.95 (0.25) | 14.27 | 1.11 | 17.48 | 12.08 (1) | 11.41 | 0.90 | 3.60 | 0.81 |
| kappa/n=0.3333 | 1e-16 | 7 | 31.90 | 6.84 | 30.40 | 30.40 (0.25) | 27.46 | 0.95 | 34.67 | 23.51 (1) | 23.28 | 0.74 | 3.44 | 0.77 |
| kappa/n=0.3333 | 1e-24 | 11 | 50.34 | 10.32 | 43.96 | 43.96 (0.25) | 40.95 | 0.87 | 52.16 | 36.38 (1) | 35.40 | 0.72 | 3.53 | 0.83 |
| kappa/n=1 | 1e-08 | 6 | 15.57 | 5.02 | 31.81 | 23.32 (0.5) | 21.57 | 1.50 | 36.03 | 21.27 (0.5) | 21.27 | 1.37 | 4.24 | 0.91 |
| kappa/n=1 | 1e-16 | 11 | 34.47 | 10.24 | 66.05 | 48.30 (0.5) | 43.38 | 1.40 | 74.62 | 43.38 (0.5) | 43.38 | 1.26 | 4.24 | 0.90 |
| kappa/n=1 | 1e-24 | 16 | n/a | n/a | 101.17 | 70.31 (0.5) | 64.39 | n/a | 114.15 | 66.09 (0.5) | 66.09 | n/a | n/a | 0.94 |
| kappa/n=4 | 1e-08 | 10 | 19.44 | 9.21 | 116.79 | 58.88 (0.5) | 58.89 | 3.03 | 120.31 | 62.57 (0.5) | 62.57 | 3.22 | 6.80 | 1.06 |
| kappa/n=4 | 1e-16 | 20 | n/a | n/a | n/a | 126.88 (0.5) | 126.90 | n/a | n/a | 134.59 (0.5) | 134.59 | n/a | n/a | 1.06 |
| kappa/n=4 | 1e-24 | 29 | n/a | n/a | n/a | 195.71 (0.5) | 195.72 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
### B4 (S): the three conventions for the count of the reshuffled mean (a sawtooth within each pass), and the seed variability of the counts (2.5%-97.5% over bootstrap resamples of the seeds)
| instance | eps | bound, i.i.d. span methods | reshuffled: first passage | stays below from | first pass end below | bootstrap range of the first passage | first / bound | stays / bound | pass end / bound | i.i.d.: first passage | stays below from | bootstrap range | reshuffled / i.i.d., range from the two bootstrap ranges |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kappa/n=0.1 | 1e-08 | 11.14 | 10.06 | 11.00 | 12.00 | [10.05, 10.07] | 0.90 | 0.99 | 1.08 | 13.33 | 13.33 | [12.25, 13.76] | [0.73, 0.82] |
| kappa/n=0.1 | 1e-16 | 29.50 | 19.08 | 20.02 | 21.00 | [19.07, 19.09] | 0.65 | 0.68 | 0.71 | 26.40 | 26.57 | [24.18, 27.05] | [0.71, 0.79] |
| kappa/n=0.1 | 1e-24 | 47.86 | 29.05 | 30.00 | 31.00 | [29.04, 29.05] | 0.61 | 0.63 | 0.65 | 39.47 | 39.47 | [37.01, 39.85] | [0.73, 0.78] |
| kappa/n=0.3333 | 1e-08 | 13.45 | 12.08 | 12.08 | 13.00 | [11.41, 12.10] | 0.90 | 0.90 | 0.97 | 14.95 | 14.95 | [14.37, 15.76] | [0.72, 0.84] |
| kappa/n=0.3333 | 1e-16 | 31.90 | 23.51 | 24.09 | 25.00 | [23.29, 24.15] | 0.74 | 0.76 | 0.78 | 30.40 | 31.55 | [28.74, 31.97] | [0.73, 0.84] |
| kappa/n=0.3333 | 1e-24 | 50.34 | 36.38 | 38.19 | 37.00 | [36.08, 39.13] | 0.72 | 0.76 | 0.74 | 43.96 | 43.96 | [41.93, 44.28] | [0.81, 0.93] |
| kappa/n=1 | 1e-08 | 15.57 | 21.27 | 21.27 | 22.00 | [21.26, 21.27] | 1.37 | 1.37 | 1.41 | 23.32 | 23.32 | [22.08, 23.34] | [0.91, 0.96] |
| kappa/n=1 | 1e-16 | 34.47 | 43.38 | 43.38 | 44.00 | [43.37, 43.38] | 1.26 | 1.26 | 1.28 | 48.30 | 48.30 | [46.83, 50.41] | [0.86, 0.93] |
| kappa/n=1 | 1e-24 | n/a | 66.09 | 66.09 | 67.00 | [66.08, 66.09] | n/a | n/a | n/a | 70.31 | 71.01 | [67.76, 71.03] | [0.93, 0.98] |
| kappa/n=4 | 1e-08 | 19.44 | 62.57 | 62.57 | 63.00 | [62.56, 62.57] | 3.22 | 3.22 | 3.24 | 58.88 | 58.88 | [58.85, 58.91] | [1.06, 1.06] |
| kappa/n=4 | 1e-16 | n/a | 134.59 | 134.59 | 135.00 | [134.59, 134.60] | n/a | n/a | n/a | 126.88 | 126.88 | [126.84, 126.93] | [1.06, 1.06] |
Comparison at equal counts: count from which the reshuffled seed mean is below the floor of inequality (expected-sweep) for span methods with i.i.d. indices at every later record (n/a: it is above the floor at the end of the range).

| instance | gamma R2 (best reshuffled step at the first eps) | below the i.i.d. floor from | until (floor above 1e-25) |
|---|---|---|---|
| kappa/n=0.1 | 2 | 10.03 | 50.94 |
| kappa/n=0.3333 | 1 | 9.06 | 53.41 |
| kappa/n=1 | 0.5 | n/a | 40.91 |
| kappa/n=4 | 0.5 | n/a | 25.75 |
### B4 (S): seed mean against expectation under i.i.d. sampling (at the count where the seed mean reaches eps, best step)
| instance | eps | i.i.d. queries t | certified lower bound on the expected error | bound / eps | same bound averaged over the seeds run | expected (seed, component) pairs never queried | realized | share of the worst seed in the seed mean, i.i.d. | reshuffled |
|---|---|---|---|---|---|---|---|---|---|
| kappa/n=0.1 | 1e-08 | 3157 | 2.28e-09 | 0.2278 | 2.62e-16 | 0.03525 | 0 | 0.3149 | 0.1263 |
| kappa/n=0.1 | 1e-16 | 6503 | 4.71e-15 | 47.11 | 1.98e-37 | 7.24e-08 | 0 | 0.9079 | 0.1642 |
| kappa/n=0.1 | 1e-24 | 9849 | 9.74e-21 | 9741 | 0 | 1.49e-13 | 0 | 0.9211 | 0.1037 |
| kappa/n=0.3333 | 1e-08 | 3572 | 4.6e-09 | 0.4597 | 2.44e-13 | 0.006946 | 0 | 0.2355 | 0.07128 |
| kappa/n=0.3333 | 1e-16 | 7527 | 9.41e-16 | 9.415 | 0 | 1.32e-09 | 0 | 0.5033 | 0.1262 |
| kappa/n=0.3333 | 1e-24 | 10998 | 1.27e-21 | 1270 | 0 | 1.66e-15 | 0 | 0.7181 | 0.5193 |
| kappa/n=1 | 1e-08 | 5714 | 1.09e-11 | 0.001093 | 2.11e-18 | 1.59e-06 | 0 | 0.2357 | 0.03231 |
| kappa/n=1 | 1e-16 | 12109 | 3.07e-22 | 3.07e-06 | 0 | 2.14e-17 | 0 | 0.35 | 0.03201 |
| kappa/n=1 | 1e-24 | 17743 | 1.56e-31 | 1.56e-07 | 0 | 5.68e-27 | 0 | 0.3837 | 0.03186 |
| kappa/n=4 | 1e-08 | 14818 | 3.37e-23 | 3.37e-15 | 0 | 5.32e-22 | 0 | 0.03288 | 0.03134 |
| kappa/n=4 | 1e-16 | 32226 | 5.51e-50 | 5.51e-34 | 0 | 1.37e-51 | 0 | 0.03359 | 0.03138 |
| kappa/n=4 | 1e-24 | 49847 | 3.56e-78 | 3.56e-54 | 0 | 1.53e-81 | 0 | 0.03393 | n/a |

## B4b: refinements of the model of B4 (two regimes, typical runs, two-term law, orthogonal extreme)

Exponents per pass of a squared error, in nats; c = gamma n lambda, g = gamma lambda = c / n. Definitions: entry 'definitions' of results/B4b_summary.json. The entry 'note' of that file holds the numbers printed in the B4 note.

### B4b (G): a mode at a fixed step g (exact second-moment matrices against the limits; crossing of the two reshuffled branches at g = 0.2384)
| g | i.i.d. limit 2(1-g)/(2-g) | i.i.d. exact, n=1000 | i.i.d. exact, n=10000 | i.i.d. exact, n=100000 | reshuffled limit min(2, log((2-g)/g)) | reshuffled E e^2 exact, n=1000 | reshuffled E e^2 exact, n=10000 | reshuffled E e^2 exact, n=100000 | reshuffled squared mean, n=1000 | reshuffled squared mean, n=10000 | reshuffled squared mean, n=100000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.9744 | 0.9746 | 0.9744 | 0.9744 | 2 | 1.974 | 1.997 | 2 | 1.974 | 1.997 | n/a |
| 0.1 | 0.9474 | 0.9476 | 0.9474 | 0.9474 | 2 | 1.987 | 1.999 | 2 | 1.987 | 1.999 | n/a |
| 0.2 | 0.8889 | 0.889 | 0.8889 | 0.8889 | 2 | 1.994 | 1.999 | 2 | 1.994 | 1.999 | n/a |
| 0.25 | 0.8571 | 0.8572 | 0.8572 | 0.8571 | 1.946 | 1.935 | 1.945 | 1.946 | 1.995 | 2 | n/a |
| 0.32 | 0.8095 | 0.8096 | 0.8095 | 0.8095 | 1.658 | 1.652 | 1.658 | 1.658 | 1.997 | 2 | n/a |
| 0.5 | 0.6667 | 0.6666 | 0.6667 | 0.6667 | 1.099 | 1.097 | 1.098 | 1.099 | 1.998 | 2 | n/a |
| 0.75 | 0.4 | 0.3998 | 0.4 | 0.4 | 0.5108 | 0.5102 | 0.5108 | 0.5108 | 1.999 | 2 | n/a |
| 0.9 | 0.1818 | 0.1817 | 0.1818 | 0.1818 | 0.2007 | 0.2005 | 0.2007 | 0.2007 | 1.999 | 2 | n/a |
| 1.1 | -0.2222 | -0.2219 | -0.2222 | -0.2222 | -0.2007 | -0.2005 | -0.2007 | -0.2007 | 2 | 2 | n/a |
| 1.25 | -0.6667 | -0.6653 | -0.6665 | -0.6667 | -0.5108 | -0.5103 | -0.5108 | -0.5108 | 2 | 2 | n/a |
| n | best c for the reshuffled order | g = c/n | largest exact exponent of E e^2 |
|---|---|---|---|
| 100 | 22.3 | 0.223 | 1.952 |
| 1000 | 236.8 | 0.2368 | 1.995 |
| 10000 | 2382 | 0.2382 | 1.999 |
### B4b (T): typical runs under i.i.d. sampling (12 seeds, fit over the passes 60-260): median [min, max] / exact expectation / heuristic / exact reshuffled exponent
| c | n=100 | n=1000 | n=10000 | n=100000 |
|---|---|---|---|---|
| 1 | 1.555 [1.431, 1.603] / 0.995 / 1.444 / 1.194 | 1.614 [1.579, 1.662] / 1.000 / 1.561 / 1.188 | 1.673 [1.618, 1.719] / 1.000 / 1.632 / 1.187 | 1.727 [1.675, 1.754] / 1.000 / 1.681 / 1.187 |
| 2 | 1.487 [1.442, 1.529] / 0.991 / 1.392 / 1.546 | 1.635 [1.550, 1.712] / 0.999 / 1.532 / 1.538 | 1.674 [1.611, 1.751] / 1.000 / 1.614 / 1.537 | 1.765 [1.615, 1.811] / 1.000 / 1.668 / 1.537 |
| 4 | 1.378 [1.316, 1.472] / 0.982 / 1.325 / 1.763 | 1.556 [1.465, 1.644] / 0.998 / 1.499 / 1.753 | 1.650 [1.585, 1.705] / 1.000 / 1.593 / 1.752 | 1.707 [1.681, 1.748] / 1.000 / 1.654 / 1.752 |
| 8 | 1.269 [1.205, 1.309] / 0.960 / 1.237 / 1.873 | 1.484 [1.399, 1.577] / 0.996 / 1.459 / 1.864 | 1.612 [1.539, 1.646] / 1.000 / 1.569 / 1.863 | 1.671 [1.622, 1.724] / 1.000 / 1.638 / 1.863 |
| 16 | 1.120 [1.090, 1.148] / 0.914 / 1.113 / 1.933 | 1.395 [1.335, 1.452] / 0.992 / 1.410 / 1.924 | 1.529 [1.487, 1.596] / 0.999 / 1.542 / 1.923 | 1.631 [1.560, 1.683] / 1.000 / 1.620 / 1.923 |
| 32 | 0.919 [0.883, 0.963] / 0.810 / 0.919 / 1.603 | 1.352 [1.306, 1.392] / 0.984 / 1.348 / 1.960 | 1.498 [1.458, 1.545] / 0.998 / 1.510 / 1.959 | 1.605 [1.584, 1.642] / 1.000 / 1.600 / 1.959 |
| 64 | n/a | 1.235 [1.185, 1.265] / 0.967 / 1.268 / 1.979 | n/a | n/a |
| 125 | n/a | 1.128 [1.112, 1.162] / 0.934 / 1.163 / 1.990 | n/a | n/a |
| 250 | n/a | 0.956 [0.942, 0.986] / 0.857 / 1.000 / 1.935 | n/a | n/a |

|heuristic - median|: max 0.04293, median 0.01296 over the 9 cells with c >= 8 and c/n <= 0.05; max 0.1106, median 0.05726 over the 12 cells with c <= 4.

### B4b (F): logistic regression, effective curvature and the two-term law min(r_rr(c_min), log((2-g)/g)), g = gamma L_eff (measured / law (R3) alone / two-term, relative errors of the two)
| problem | L_eff / R2 | L_eff / mean_i L_i | L_eff / lmax(H*) | gamma R2 where the two terms cross | two-term exponent there | best step of the grid (B4) | its measured exponent |
|---|---|---|---|---|---|---|---|
| kappa=10, alpha=0 | 0.164 | 0.2638 | 1.417 | 1.465 | 1.991 | 0.5 | 1.987 |
| kappa=10, alpha=1 | 0.214 | 0.5718 | 1.35 | 1.126 | 1.988 | 1 | 2.014 |
| kappa=10, alpha=2 | 0.2304 | 1.056 | 1.356 | 1.047 | 1.987 | 0.5 | 1.993 |
| kappa=100, alpha=0 | 0.2804 | 0.5596 | 11.35 | 0.9142 | 1.917 | 0.5 | 1.885 |
| kappa=100, alpha=1 | 0.2326 | 0.8137 | 3.38 | 1.116 | 1.903 | 1 | 1.88 |
| kappa=100, alpha=2 | 0.1756 | 1.478 | 2.404 | 1.46 | 1.917 | 1 | 1.926 |
| kappa=1000, alpha=0 | 0.3962 | 1.229 | 35.41 | 0.8175 | 1.644 | 0.5 | 1.445 |
| kappa=1000, alpha=1 | 0.2517 | 1.234 | 5.759 | 1.34 | 1.596 | 1 | 1.542 |
| kappa=1000, alpha=2 | 0.1807 | 2.136 | 3.581 | 1.959 | 1.536 | 2 | 1.493 |
| kappa=10000, alpha=0 | 0.4257 | 1.816 | 53.17 | 1.071 | 1.219 | 1 | 1.006 |
| kappa=10000, alpha=1 | 0.2658 | 1.856 | 8.863 | 1.834 | 1.133 | 2 | 0.8414 |
| kappa=10000, alpha=2 | 0.1869 | 2.981 | 4.979 | 3.614 | 0.6731 | 2 | 0.4327 |
| problem | gamma R2 = 0.007812 | gamma R2 = 0.01562 | gamma R2 = 0.03125 | gamma R2 = 0.0625 | gamma R2 = 0.125 | gamma R2 = 0.25 | gamma R2 = 0.5 | gamma R2 = 1 | gamma R2 = 2 |
|---|---|---|---|---|---|---|---|---|---|
| kappa=10, alpha=0 | 1.102 / 1.078 / 1.078 (+2%, +2%) | 1.477 / 1.457 / 1.457 (+1%, +1%) | 1.719 / 1.708 / 1.708 (+1%, +1%) | 1.843 / 1.840 / 1.840 (+0%, +0%) | 1.924 / 1.910 / 1.910 (+1%, +1%) | 1.954 / 1.951 / 1.951 (+0%, +0%) | 1.987 / 1.974 / 1.974 (+1%, +1%) | 1.986 / 1.987 / 1.987 (-0%, -0%) | 1.637 / 1.993 / 1.629 (-18%, +1%) |
| kappa=10, alpha=1 | 1.076 / 1.047 / 1.047 (+3%, +3%) | 1.462 / 1.432 / 1.432 (+2%, +2%) | 1.714 / 1.693 / 1.693 (+1%, +1%) | 1.836 / 1.833 / 1.833 (+0%, +0%) | 1.925 / 1.906 / 1.906 (+1%, +1%) | 1.956 / 1.949 / 1.949 (+0%, +0%) | 1.987 / 1.973 / 1.973 (+1%, +1%) | 2.014 / 1.986 / 1.986 (+1%, +1%) | 1.316 / 1.993 / 1.301 (-34%, +1%) |
| kappa=10, alpha=2 | 1.075 / 1.041 / 1.041 (+3%, +3%) | 1.471 / 1.427 / 1.427 (+3%, +3%) | 1.735 / 1.691 / 1.691 (+3%, +3%) | 1.852 / 1.832 / 1.832 (+1%, +1%) | 1.932 / 1.905 / 1.905 (+1%, +1%) | 1.961 / 1.948 / 1.948 (+1%, +1%) | 1.993 / 1.973 / 1.973 (+1%, +1%) | 1.907 / 1.986 / 1.986 (-4%, -4%) | 1.215 / 1.993 / 1.206 (-39%, +1%) |
| kappa=100, alpha=0 | 0.245 / 0.238 / 0.238 (+3%, +3%) | 0.460 / 0.447 / 0.447 (+3%, +3%) | 0.802 / 0.780 / 0.780 (+3%, +3%) | 1.218 / 1.191 / 1.191 (+2%, +2%) | 1.565 / 1.539 / 1.539 (+2%, +2%) | 1.770 / 1.754 / 1.754 (+1%, +1%) | 1.885 / 1.863 / 1.863 (+1%, +1%) | 1.695 / 1.923 / 1.814 (-12%, -7%) | 0.908 / 1.959 / 0.942 (-54%, -4%) |
| kappa=100, alpha=1 | 0.176 / 0.166 / 0.166 (+6%, +6%) | 0.338 / 0.318 / 0.318 (+6%, +6%) | 0.617 / 0.582 / 0.582 (+6%, +6%) | 1.014 / 0.964 / 0.964 (+5%, +5%) | 1.420 / 1.363 / 1.363 (+4%, +4%) | 1.710 / 1.652 / 1.652 (+4%, +4%) | 1.852 / 1.812 / 1.812 (+2%, +2%) | 1.880 / 1.894 / 1.894 (-1%, -1%) | 1.175 / 1.942 / 1.194 (-39%, -2%) |
| kappa=100, alpha=2 | 0.157 / 0.151 / 0.151 (+4%, +4%) | 0.303 / 0.292 / 0.292 (+4%, +4%) | 0.560 / 0.539 / 0.539 (+4%, +4%) | 0.949 / 0.908 / 0.908 (+4%, +4%) | 1.387 / 1.314 / 1.314 (+6%, +6%) | 1.704 / 1.622 / 1.622 (+5%, +5%) | 1.851 / 1.797 / 1.797 (+3%, +3%) | 1.926 / 1.886 / 1.886 (+2%, +2%) | 1.585 / 1.937 / 1.546 (-18%, +2%) |
| kappa=1000, alpha=0 | n/a | 0.100 / 0.100 / 0.100 (+0%, +0%) | 0.196 / 0.196 / 0.196 (-0%, -0%) | 0.372 / 0.372 / 0.372 (-0%, -0%) | 0.668 / 0.669 / 0.669 (-0%, -0%) | 1.065 / 1.069 / 1.069 (-0%, -0%) | 1.445 / 1.449 / 1.449 (-0%, -0%) | 1.176 / 1.703 / 1.398 (-31%, -16%) | 0.303 / 1.838 / 0.421 (-84%, -28%) |
| kappa=1000, alpha=1 | n/a | n/a | 0.115 / 0.106 / 0.106 (+9%, +9%) | 0.220 / 0.207 / 0.207 (+6%, +6%) | 0.417 / 0.393 / 0.393 (+6%, +6%) | 0.745 / 0.700 / 0.700 (+6%, +6%) | 1.171 / 1.104 / 1.104 (+6%, +6%) | 1.542 / 1.476 / 1.476 (+4%, +4%) | 0.989 / 1.719 / 1.089 (-42%, -9%) |
| kappa=1000, alpha=2 | n/a | n/a | n/a | 0.130 / 0.124 / 0.124 (+5%, +5%) | 0.251 / 0.241 / 0.241 (+4%, +4%) | 0.471 / 0.452 / 0.452 (+4%, +4%) | 0.821 / 0.788 / 0.788 (+4%, +4%) | 1.250 / 1.199 / 1.199 (+4%, +4%) | 1.493 / 1.545 / 1.511 (-3%, -1%) |
| kappa=10000, alpha=0 | n/a | n/a | n/a | 0.120 / 0.120 / 0.120 (-0%, -0%) | 0.233 / 0.233 / 0.233 (-0%, -0%) | 0.438 / 0.439 / 0.439 (-0%, -0%) | 0.764 / 0.769 / 0.769 (-1%, -1%) | 1.006 / 1.179 / 1.179 (-15%, -15%) | 0.131 / 1.531 / 0.299 (-91%, -56%) |
| kappa=10000, alpha=1 | n/a | n/a | n/a | n/a | 0.121 / 0.121 / 0.121 (-0%, -0%) | 0.235 / 0.235 / 0.235 (-0%, -0%) | 0.441 / 0.442 / 0.442 (-0%, -0%) | 0.770 / 0.774 / 0.774 (-1%, -1%) | 0.841 / 1.184 / 1.016 (-29%, -17%) |
| kappa=10000, alpha=2 | n/a | n/a | n/a | n/a | n/a | n/a | 0.119 / 0.111 / 0.111 (+7%, +7%) | 0.229 / 0.217 / 0.217 (+5%, +5%) | 0.433 / 0.411 / 0.411 (+5%, +5%) |
| group of (problem, step) pairs | pairs | law (R3): max |rel. err.| | median | within 10% | two-term law: max |rel. err.| | median | within 10% |
|---|---|---|---|---|---|---|---|
| gammaR2_le_half | 65 | 0.08647 | 0.02171 | 65 | 0.08647 | 0.02171 | 65 |
| gammaR2_ge_1 | 24 | 0.9145 | 0.1328 | 11 | 0.5627 | 0.0379 | 19 |
| gammaR2_ge_1_kappa_le_100 | 8 | 0.5363 | 0.1488 | 3 | 0.06565 | 0.01274 | 8 |
| gammaR2_ge_1_kappa_ge_1000 | 10 | 0.9145 | 0.05397 | 6 | 0.5627 | 0.05397 | 7 |
| all | 89 | 0.9145 | 0.02991 | 76 | 0.5627 | 0.02268 | 84 |
i.i.d. runs: measured median exponent / 1 - g/(2-g) (the heuristic for the EXPECTATION at large c):

| problem | gamma R2 = 0.007812 | gamma R2 = 0.01562 | gamma R2 = 0.03125 | gamma R2 = 0.0625 | gamma R2 = 0.125 | gamma R2 = 0.25 | gamma R2 = 0.5 | gamma R2 = 1 | gamma R2 = 2 |
|---|---|---|---|---|---|---|---|---|---|
| kappa=10, alpha=0 | 1.523 / 0.999 | 1.611 / 0.999 | 1.496 / 0.997 | 1.429 / 0.995 | 1.293 / 0.990 | 1.271 / 0.979 | 1.159 / 0.957 | 1.004 / 0.911 | 0.888 / 0.804 |
| kappa=10, alpha=1 | 1.490 / 0.999 | 1.611 / 0.998 | 1.384 / 0.997 | 1.425 / 0.993 | 1.270 / 0.986 | 1.262 / 0.973 | 1.172 / 0.943 | 0.962 / 0.880 | 0.798 / 0.728 |
| kappa=10, alpha=2 | 1.504 / 0.999 | 1.471 / 0.998 | 1.404 / 0.996 | 1.347 / 0.993 | 1.311 / 0.985 | 1.303 / 0.970 | 1.143 / 0.939 | 0.959 / 0.870 | 0.754 / 0.701 |
| kappa=100, alpha=0 | 0.259 / 0.999 | 0.520 / 0.998 | 1.030 / 0.996 | 1.336 / 0.991 | 1.274 / 0.982 | 1.133 / 0.964 | 1.023 / 0.925 | 0.871 / 0.837 | 0.582 / 0.610 |
| kappa=100, alpha=1 | 0.183 / 0.999 | 0.366 / 0.998 | 0.735 / 0.996 | 1.390 / 0.993 | 1.358 / 0.985 | 1.264 / 0.970 | 1.128 / 0.938 | 0.963 / 0.868 | 0.723 / 0.697 |
| kappa=100, alpha=2 | 0.163 / 0.999 | 0.325 / 0.999 | 0.651 / 0.997 | 1.295 / 0.994 | 1.393 / 0.989 | 1.299 / 0.978 | 1.221 / 0.954 | 1.007 / 0.904 | 0.876 / 0.787 |
| kappa=1000, alpha=0 | n/a | 0.102 / 0.997 | 0.205 / 0.994 | 0.410 / 0.987 | 0.821 / 0.975 | 1.061 / 0.948 | 0.913 / 0.890 | 0.717 / 0.753 | 0.231 / 0.344 |
| kappa=1000, alpha=1 | n/a | n/a | 0.118 / 0.996 | 0.230 / 0.992 | 0.462 / 0.984 | 0.911 / 0.968 | 1.059 / 0.933 | 0.892 / 0.856 | 0.626 / 0.664 |
| kappa=1000, alpha=2 | n/a | n/a | 0.069 / 0.997 | 0.133 / 0.994 | 0.266 / 0.989 | 0.532 / 0.977 | 1.062 / 0.953 | 1.072 / 0.901 | 0.854 / 0.779 |
| kappa=10000, alpha=0 | n/a | n/a | n/a | 0.123 / 0.987 | 0.247 / 0.973 | 0.494 / 0.944 | 0.931 / 0.881 | 0.673 / 0.730 | n/a |
| kappa=10000, alpha=1 | n/a | n/a | n/a | n/a | 0.124 / 0.983 | 0.248 / 0.966 | 0.498 / 0.929 | 0.895 / 0.847 | 0.570 / 0.638 |
| kappa=10000, alpha=2 | n/a | n/a | n/a | n/a | n/a | n/a | 0.122 / 0.951 | 0.241 / 0.897 | 0.481 / 0.770 |
### B4b (O): components on orthogonal directions (n = 1000, 40000 passes; c = gamma n lambda = gamma R2)
| c | fixed order, n -> infinity (exact) | fixed order, exact at n | reshuffled, typical run: median [min, max] | i.i.d., typical run: median [min, max] |
|---|---|---|---|---|
| 0.05 | 0.09734 | 0.09735 | 0.0978 [0.0978, 0.0978] | 0.1001 [0.1000, 0.1001] |
| 0.1 | 0.1888 | 0.1889 | 0.1907 [0.1907, 0.1907] | 0.2005 [0.2003, 0.2007] |
| 0.25 | 0.4239 | 0.424 | 0.4358 [0.4357, 0.4358] | 0.4962 [0.4956, 0.4965] |
| 0.4 | 0.6014 | 0.6016 | 0.6324 [0.6320, 0.6326] | 0.6524 [0.6505, 0.6577] |
| 0.5 | 0.6931 | 0.6934 | 0.6939 [0.6937, 0.6940] | 0.5850 [0.5829, 0.5905] |
| 0.6 | 0.2533 | 0.255 | 0.3080 [0.3069, 0.3086] | 0.3548 [0.3521, 0.3554] |
| 0.65 | 0.06079 | 0.0624 | 0.1134 [0.1124, 0.1135] | 0.1937 [0.1906, 0.1951] |
| 0.7 | -0.117 | -0.1154 | -0.0653 [-0.0660, -0.0651] | 0.0234 [0.0232, 0.0270] |
| 1 | -0.9624 | -0.961 | -0.9111 [-0.9117, -0.9111] | -0.8118 [-0.8145, -0.8100] |
