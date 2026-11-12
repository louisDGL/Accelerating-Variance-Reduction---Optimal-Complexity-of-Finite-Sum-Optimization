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
