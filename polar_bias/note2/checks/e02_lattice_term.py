"""Second-order lattice term for Euclidean normalisation in R^2 (codimension 2), law of e01.
Prediction: sqrt(b)*(b*G_b - g*) -> -(sqrt2/(4 pi J)) * E_Lambda(g*)_1,  J = sqrt(pi/2)/2,
E_Lambda(m) = lim_tau [ 9 sum_{l in Lambda} o(m+l) e^{-|l|^2/2tau^2} - int o(m+y) e^{-|y|^2/2tau^2} dy ],
Lambda = -b(1,1) + 3 Z^2 (depends on b mod 3), int = 2 pi tau J m (1+O(tau^-2))."""
import numpy as np
J = np.sqrt(np.pi/2)/2; g = 0.125
for cls, shift in [(0, 0.0), (1, -1.0), (2, -2.0)]:
    out = []
    for tau in [100.0, 200.0, 400.0]:
        n = int(6*tau/3)
        k = 3.0*np.arange(-n, n+1)+shift
        X, Y = np.meshgrid(k, k, indexing='ij')
        w = np.exp(-(X**2+Y**2)/(2*tau**2))
        x = X+g; y = Y+g; r = np.hypot(x, y); r[r == 0] = 1
        s = 9*np.sum(w*x/r)
        # exact Gaussian integral: 2 pi tau^2 * Phi_{tau^2 I}(m)_1, Phi(m) = E (m+tau Z)/|m+tau Z|
        from scipy.special import ive
        a = (np.sqrt(2)*g)**2/(4*tau**2)   # |m|^2/(4 tau^2), |m| = sqrt2 g
        # E o(m+tau Z) = m/|m| * sqrt(pi/2) * (|m|/(2 tau)) ... use closed form: sqrt(pi a) e^{-a}(I0(a)+I1(a)) * m/|m|, a=|m|^2/(4tau^2)
        phi = np.sqrt(np.pi*a/2)*(ive(0, a)+ive(1, a))/np.sqrt(2)   # component along e1 of unit vector (1,1)/sqrt2
        integ = 2*np.pi*tau**2*phi
        out.append(s-integ)
    print(f"b mod 3 = {cls}: E_Lambda(g*)_1 for tau=100,200,400: {out[0]:+.4f} {out[1]:+.4f} {out[2]:+.4f};  predicted sqrt(b)*err -> {-np.sqrt(2)/(4*np.pi*J)*out[-1]:+.4f}")
