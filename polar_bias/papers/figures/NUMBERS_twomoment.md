# Numbers behind fig_constants, fig_law, fig_digits (two-moment paper)

All "+-" are one Monte-Carlo standard error unless stated. Scripts: `papers/experiments/` (figA_*, figB_*, digits_*, figC_plot, make_numbers).

## FIG A - constants (exact, de Bruijn Pfaffian formula of note2/a1/wish.py; precision raised until E tr W^2 = Kd(K+d+1) holds to 1e-25)

Cross-checks: q_(2,2) = 0.320377 = (3 sqrt2 - 2)/7 = 0.320377 (16 digits agree; MC 0.3204); q_(3,5) = 0.842571 (MC 0.8425); all 15 Monte-Carlo values of results/c02_constants.json agree to their 3-4 digits.

| K | d | q | a1 | a2 | a1+a2 | lower 3/(K+d+1) | upper 3/(max+2) | max*(a1+a2) |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 | 0.320377 | -0.413207 | 1.09585 | 0.682641 | 0.6 | 0.75 | 1.36528 |
| 2 | 3 | 0.393398 | -0.204058 | 0.772078 | 0.568019 | 0.5 | 0.6 | 1.70406 |
| 3 | 3 | 0.686882 | -0.168491 | 0.693342 | 0.524852 | 0.428571 | 0.6 | 1.57456 |
| 3 | 5 | 0.842571 | -0.0767815 | 0.478365 | 0.401584 | 0.333333 | 0.428571 | 2.00792 |
| 4 | 4 | 1.06793 | -0.0946553 | 0.522644 | 0.427989 | 0.333333 | 0.5 | 1.71195 |
| 8 | 8 | 2.635 | -0.024851 | 0.272951 | 0.2481 | 0.176471 | 0.3 | 1.9848 |
| 16 | 16 | 5.81487 | -0.0065342 | 0.141995 | 0.13546 | 0.0909091 | 0.166667 | 2.16737 |
| 64 | 64 | 24.9957 | -0.000430313 | 0.0369258 | 0.0364954 | 0.0232558 | 0.0454545 | 2.33571 |
| 8 | 32 | 3.35084 | -0.00234896 | 0.087952 | 0.085603 | 0.0731707 | 0.0882353 | 2.7393 |
| 32 | 128 | 14.94 | -0.000150622 | 0.0224674 | 0.0223168 | 0.0186335 | 0.0230769 | 2.85655 |

(128,128) NOT computed (interrupted; a single size needs > 10 min). Largest sizes done: square 96x96, d=2K 64x128, d=4K 64x256.

a1 < 0 < a2 and 3/(K+d+1) < a1+a2 < 3/(max+2): hold at all 34 plotted sizes.

| ratio | limit 6 kappa(gamma) | largest size | max*(a1+a2) there | gap to limit |
|---|---|---|---|---|
| d=1K | 2.400000 | 96x96 | 2.35660 | 0.0434 |
| d=2K | 2.775401 | 64x128 | 2.73594 | 0.0395 |
| d=4K | 2.898092 | 64x256 | 2.87716 | 0.0209 |

Marchenko-Pastur limits by numerical quadrature (density normalisation checked: 1.00000000, 1.00000000, 1.00000000); 12/5 = 2.4 reproduced to 1e-6. Approach to the limit is from below and slow (gap ~ O(1/max)).

## FIG B(a) - law across noise laws, 2x3: ratio <b G_b, g*>/|g*|_F^2 (zero of the frozen mean field)

Blocks of 2e+06 batch means (1e6 for rank-one laws at b >= 256), zero solved per block, se = block-to-block. Predictions use EXACT cumulants and exact q_(2,3) = 0.393398, a1 = -0.20406, a2 = 0.77208; Kronecker law: J, T by Monte Carlo (4 x 4e6), |se(g*)|_F = 0.0026 on |g*|_F = 1.363.

| law | |g*|_F | b=1 | b=4 | b=16 | b=64 | b=256 | b=1024 | pooled b>=64 |
|---|---|---|---|---|---|---|---|---|
| iid | 0.4638 | 0.935+-0.001 | 0.985+-0.001 | 0.996+-0.002 | 0.990+-0.004 | 0.991+-0.008 | 0.986+-0.010 | 0.990+-0.003 |
| coskew | 0.0803 | 0.509+-0.002 | 0.851+-0.007 | 0.975+-0.010 | 0.978+-0.027 | 1.021+-0.028 | 0.943+-0.036 | 0.986+-0.017 |
| factor | 0.0442 | 0.782+-0.005 | 0.926+-0.016 | 0.939+-0.042 | 0.937+-0.023 | 0.942+-0.039 | 0.944+-0.059 | 0.939+-0.019 |
| rank1 | 0.9276 | - | -0.021+-0.000 | 0.320+-0.001 | 0.691+-0.002 | 0.892+-0.006 | 0.986+-0.011 | 0.722+-0.002 |
| kron | 1.3628 | - | 0.003+-0.000 | 0.373+-0.001 | 0.732+-0.004 | 0.897+-0.020 | 0.964+-0.025 | 0.745+-0.004 |

Cosine between b G_b and g* at b = 1024: iid 0.9995, coskew 0.9974, factor 0.9913, rank1 0.9998, kron 0.9998.

Not as predicted / caveats: (i) rank-one laws converge slowly (0.69-0.73 at b=64, 0.89-0.90 at b=256, 0.96-0.99 at b=1024) and the isotropic rank-one law has the WRONG SIGN at b=4 (-0.021+-0.0004); (ii) "one skewed factor" plateaus at 0.94 (pooled b>=16: 0.939+-0.017, about 3.5 se below 1) with block-to-block scatter across b much smaller than the quoted se (chi2 = 0.02 on 3 dof) - unresolved; (iii) i.i.d. law sits at 0.990+-0.003 for b>=64 (3 se below 1). (ii)-(iii) are percent-level and may be statistical, but they were not tuned away.

## FIG B(b) - momentum (frozen stationary filtered noise, i.i.d. Exp(1)-1 entries, 2x3), g* per entry = 0.18934

| beta | b | b_eff | G b_eff/g* (filled) | G b(1+beta)/(1-beta)/g* (hollow) | samples |
|---|---|---|---|---|---|
| 0.0 | 1 | 1.00 | 0.9356+-0.0009 | 0.9356+-0.0009 | 4e+06 |
| 0.0 | 4 | 4.00 | 0.9844+-0.0015 | 0.9844+-0.0015 | 6e+06 |
| 0.0 | 16 | 16.00 | 0.9926+-0.0029 | 0.9926+-0.0029 | 1e+07 |
| 0.5 | 1 | 2.33 | 0.9180+-0.0007 | 1.1802+-0.0009 | 4e+06 |
| 0.5 | 4 | 9.33 | 0.9814+-0.0036 | 1.2618+-0.0047 | 6e+06 |
| 0.5 | 16 | 37.33 | 0.9933+-0.0034 | 1.2771+-0.0043 | 1e+07 |
| 0.9 | 1 | 14.26 | 0.9798+-0.0025 | 1.3052+-0.0033 | 4e+06 |
| 0.9 | 4 | 57.05 | 0.9998+-0.0072 | 1.3318+-0.0096 | 6e+06 |
| 0.9 | 16 | 228.21 | 0.9960+-0.0083 | 1.3268+-0.0110 | 1e+07 |
| 0.95 | 1 | 29.26 | 0.9878+-0.0081 | 1.3167+-0.0108 | 4e+06 |
| 0.95 | 4 | 117.03 | 1.0096+-0.0031 | 1.3459+-0.0042 | 6e+06 |
| 0.95 | 16 | 468.10 | 0.9919+-0.0127 | 1.3222+-0.0169 | 1e+07 |

Skewness-effective normalisation gives 1 within ~1% for b >= 4 (0.92-0.99 at b = 1, a finite-b effect). The variance-effective normalisation gives (1+beta)^2/(1+beta+beta^2) = 1.286 (0.5), 1.332 (0.9), 1.333 (0.95), i.e. ABOVE 1 (the brief expected "0.75-0.78": that is the reciprocal, prediction/measurement). Consistent with results/c04_frozen.json (0.917, 0.997, 0.974, 1.017, 1.008).

## FIG B(c) - sign-blind co-skewness law: b (G_b)_11, prediction polar 0.08033, sign descent 0

| b | polar | sign (-b median) |
|---|---|---|
| 1 | 0.0409+-0.0001 | -0.0566+-0.0002 |
| 4 | 0.0684+-0.0006 | -0.0208+-0.0006 |
| 16 | 0.0783+-0.0008 | -0.0044+-0.0009 |
| 64 | 0.0786+-0.0022 | -0.0030+-0.0024 |
| 256 | 0.0821+-0.0023 | -0.0026+-0.0028 |
| 1024 | 0.0757+-0.0029 | -0.0056+-0.0033 |

Sign step: -0.057 at b=1, -0.021 at b=4, -0.0044+-0.0009 at b=16 (4.7 se: a higher-order residue decaying faster than 1/b), compatible with 0 for b >= 64; polar step within 1-1.6 se of the prediction for b >= 16.

## FIG C - digits least squares (real data), prediction (5), no fitted parameter

Instance: sklearn load_digits, n = 1797, d = 16 leading principal components of standardised pixels (centred, not whitened), K = 4 fixed random orthonormal projections (default_rng(0)) of the centred one-hot labels, lam = 0.001; N = 64. F(0)-F* = 0.0901. Hessian eigenvalues 1.17..7.34.
WHY NOT one-hot: K=10 gives a singular gradient covariance (16-dim kernel, min eig -2e-16); K=9 (drop a class) is nondegenerate but eig ratio 1.3e-4 and g*^T Sigma^-1 g* ~ 90 (pilot, 4e5 samples), so b <= 128 is far from first order; the K=4 projection was chosen BEFORE any simulation on the criterion g*^T Sigma^-1 g* (26 here).
Gradient covariance at W*: eig min/max = 3.645e-03/0.809 = 4.51e-03.
Prediction (8 independent MC evaluations of J, T with 2e+06 Gaussian samples each, linear control variate): ||g*||_* = 3.8010 +- 0.0046 (sd over seeds 0.0130 = 0.34%); |g*|_F = 2.4601 (seed sd 0.0082); <g*,H^-1 g*>/2 = 1.6145 +- 0.0038 (i.e. <g*,H^-1 g*> = 3.2290 +- 0.0075); g*^T Sigma^-1 g* = 26.2.

Runs: batches with replacement, exact polar (batched SVD), eta_t = 0.02 (1+t/300)^-0.75, start W = 0, tail = last third split in two halves; R and T in the table. se = jackknife over runs.

| beta | b | b_eff | R | T | b_eff ||grad F||_* | ratio to ||g*||_* | halves (1st/2nd) | excess loss | predicted | ratio | halves ratio 2nd/1st | cos | <b_eff G, g*>/|g*|^2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 4 | 4.0 | 64 | 3e+05 | 1.726+-0.002 | 0.454 | 1.726/1.727 | 1.616e-02+-3.6e-05 | 1.009e-01 | 0.160 | 1.001 | 0.8915+-0.0003 | 0.381+-0.000 |
| 0.0 | 8 | 8.0 | 64 | 3e+05 | 2.819+-0.003 | 0.742 | 2.819/2.820 | 1.334e-02+-2.9e-05 | 2.523e-02 | 0.529 | 1.003 | 0.9443+-0.0001 | 0.724+-0.001 |
| 0.0 | 16 | 16.0 | 64 | 3e+05 | 4.149+-0.005 | 1.092 | 4.149/4.149 | 8.060e-03+-2.0e-05 | 6.307e-03 | 1.278 | 1.001 | 0.9563+-0.0001 | 1.132+-0.001 |
| 0.0 | 32 | 32.0 | 64 | 3e+05 | 5.625+-0.006 | 1.480 | 5.623/5.627 | 4.063e-03+-9.7e-06 | 1.577e-03 | 2.577 | 1.003 | 0.9610+-0.0001 | 1.604+-0.002 |
| 0.0 | 64 | 64.0 | 64 | 3e+05 | 6.854+-0.008 | 1.803 | 6.860/6.848 | 1.597e-03+-4.1e-06 | 3.942e-04 | 4.051 | 0.996 | 0.9665+-0.0001 | 2.004+-0.003 |
| 0.0 | 128 | 128.0 | 64 | 3e+05 | 7.359+-0.011 | 1.936 | 7.332/7.387 | 4.746e-04+-1.6e-06 | 9.854e-05 | 4.816 | 1.016 | 0.9725+-0.0001 | 2.183+-0.004 |
| 0.0 | 512 | 512.0 | 32 | 3e+05 | 5.572+-0.018 | 1.466 | 5.572/5.574 | 1.584e-05+-1.0e-07 | 6.159e-06 | 2.571 | 1.006 | 0.9835+-0.0004 | 1.599+-0.005 |
| 0.9 | 4 | 57.1 | 64 | 3e+05 | 3.619+-0.026 | 0.952 | 3.627/3.614 | 4.589e-04+-8.0e-06 | 4.960e-04 | 0.925 | 0.995 | 0.9859+-0.0004 | 0.968+-0.008 |
| 0.9 | 8 | 114.1 | 64 | 3e+05 | 4.264+-0.026 | 1.122 | 4.243/4.290 | 1.685e-04+-2.3e-06 | 1.240e-04 | 1.359 | 1.010 | 0.9902+-0.0004 | 1.173+-0.008 |
| 0.9 | 16 | 228.2 | 64 | 3e+05 | 4.773+-0.036 | 1.256 | 4.823/4.729 | 5.407e-05+-8.8e-07 | 3.100e-05 | 1.744 | 0.955 | 0.9930+-0.0005 | 1.327+-0.011 |
| 0.9 | 32 | 456.4 | 64 | 3e+05 | 4.902+-0.045 | 1.290 | 4.865/4.948 | 1.440e-05+-2.6e-07 | 7.750e-06 | 1.858 | 1.027 | 0.9942+-0.0006 | 1.366+-0.013 |
| 0.9 | 64 | 912.8 | 64 | 3e+05 | 4.677+-0.064 | 1.230 | 4.818/4.559 | 3.153e-06+-8.0e-08 | 1.938e-06 | 1.627 | 0.888 | 0.9948+-0.0010 | 1.277+-0.017 |
| 0.9 | 128 | 1825.7 | 64 | 3e+05 | 4.357+-0.069 | 1.146 | 4.343/4.408 | 6.767e-07+-2.1e-08 | 4.844e-07 | 1.397 | 1.033 | 0.9957+-0.0015 | 1.178+-0.019 |

Equilibration: first and second half of the tail agree within the jackknife se for beta = 0 (differences <= 0.8% in the norm, <= 1.6% in the loss) and within ~2 se for beta = 0.9 (largest: b=64, 4.82 vs 4.56, 5.5% in norm, 12% in loss, se of each half ~0.09) - no systematic drift direction. Noise-debiased excess loss (U-statistic over runs) differs from the plain one by < 0.4% everywhere.
Step-size check (eta0/4 = 0.005, same batch seeds): beta=0.0, b=128: b_eff||grad||_* = 7.359+-0.011 vs 7.359+-0.011; excess 4.747e-04 vs 4.746e-04.
Step-size check (eta0/4 = 0.005, same batch seeds): beta=0.9, b=16: b_eff||grad||_* = 4.630+-0.035 vs 4.773+-0.036; excess 5.050e-05 vs 5.407e-05.

NOT AS PREDICTED: the direction is right (cos 0.89 -> 0.983 for beta=0, 0.986 -> 0.996 for beta=0.9) but the magnitude has not converged. beta = 0: b ||grad F||_* rises from 0.45 to 1.94 times ||g*||_* at b = 128 (7.36 vs 3.80) and comes back to 1.47 at b = 512; excess loss is 0.16x (b=4) to 4.8x (b=128) the prediction, 2.6x at b=512. beta = 0.9: 0.95, 1.12, 1.26, 1.29, 1.23, 1.15 times ||g*||_* for b = 4..128 (non-monotone, peak at b_eff ~ 456, decreasing towards the prediction); excess loss 0.93x-1.86x. The two beta do not collapse on b_eff at fixed b_eff (7.36 vs 4.26 near b_eff ~ 120), so the excess is a higher-order (finite-b) effect with a different momentum dependence, not a wrong b_eff; it is equilibrated and for beta=0 insensitive to the step size (b=128: identical at eta0/4). For beta=0.9 there is a residual step-size effect: at b=16, eta0/4 lowers the value by 3% (4.63 vs 4.77; same random batches, so the difference is real), i.e. towards the prediction but far from closing the 26% gap. A b = 2048 run was interrupted and is not available, so convergence of the beta = 0 curve to ||g*||_* is NOT demonstrated, only its turn-over.

## Not done / compute notes

(128,128) constants and sizes with min(K,d) > 96 (square) not computed; digits b = 2048 (beta=0) interrupted at 50%; no b > 128 for beta = 0.9. Total compute exceeded the 45-minute budget (two machine crashes, shared machine); the last sequential law job peaked at ~5 GB RSS.

