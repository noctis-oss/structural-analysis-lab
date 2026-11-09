"""Lab 6 · displacements and rotations. Every number printed in the lab, by two independent routes:
(1) the hand formulas of the lab (double integration closed forms, virtual-work integrals evaluated numerically);
(2) a direct-stiffness model (Euler-Bernoulli beam elements with exact consistent loads; pin-jointed truss), which is
    what Noctis Effect's Linear elastic mode does.
Units: kN, m. Signs: deflection v up +, rotation counterclockwise +."""
import numpy as np

def simpson(f, a, b, n=2000):
    x = np.linspace(a, b, 2 * n + 1); y = f(x); h = (b - a) / (2 * n)
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

# ---------------------------------------------------------------- beam by direct stiffness
def beam_fe(L, EI, supports, points=(), udl=(), nel=240):
    """supports: list of (x, 'fixed'|'pin'|'roller'); points: (x, P up +); udl: (a, b, w up +).
    Element nodes include every support, load point and udl end. Returns (xs, v, theta, reactions)."""
    keys = {0.0, float(L)} | {float(x) for x, _ in supports} | {float(x) for x, _ in points}
    for a, b, _ in udl: keys |= {float(a), float(b)}
    xs = np.unique(np.round(np.concatenate([np.linspace(0, L, nel + 1), sorted(keys)]), 9))
    n = len(xs); K = np.zeros((2 * n, 2 * n)); F = np.zeros(2 * n)
    for e in range(n - 1):
        x1, x2 = xs[e], xs[e + 1]; h = x2 - x1
        k = EI / h**3 * np.array([[12, 6*h, -12, 6*h], [6*h, 4*h*h, -6*h, 2*h*h],
                                  [-12, -6*h, 12, -6*h], [6*h, 2*h*h, -6*h, 4*h*h]])
        d = [2*e, 2*e+1, 2*e+2, 2*e+3]; K[np.ix_(d, d)] += k
        for a, b, w in udl:
            if x1 >= a - 1e-12 and x2 <= b + 1e-12:
                F[d] += w * np.array([h/2, h*h/12, h/2, -h*h/12])
    for x, P in points:
        F[2 * int(np.argmin(abs(xs - x)))] += P
    fixed = []
    for x, t in supports:
        i = int(np.argmin(abs(xs - x))); fixed.append(2 * i)
        if t == 'fixed': fixed.append(2 * i + 1)
    free = [i for i in range(2 * n) if i not in fixed]
    u = np.zeros(2 * n); u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    R = K @ u - F
    return xs, u[0::2], u[1::2], {i: R[i] for i in fixed}

def at(xs, arr, x): return float(np.interp(x, xs, arr))

out = []
def show(label, value, unit='', fmt='{:.6g}'):
    out.append(f'{label:58s} {fmt.format(value)} {unit}')

E = 200e6                                   # kN/m^2  (200 GPa)

# ---------------------------------------------------------------- Example 6.1 · cantilever, tip load
L, P, I = 3.0, 20.0, 60e-6                  # m, kN, m^4 (60 x 10^6 mm^4)
EI = E * I                                  # 12 000 kN m^2
show('6.1 EI', EI, 'kN m2')
th_B = -P * L**2 / (2 * EI); v_B = -P * L**3 / (3 * EI)
v_mid = -P * 1.5**2 * (3 * L - 1.5) / (6 * EI)
show('6.1 hand theta_B = -PL^2/2EI', th_B, 'rad'); show('6.1 hand v_B = -PL^3/3EI', v_B * 1e3, 'mm')
show('6.1 hand v(1.5)', v_mid * 1e3, 'mm')
xs, v, th, R = beam_fe(L, EI, [(0, 'fixed')], points=[(L, -P)])
show('6.1 FE   theta_B', at(xs, th, L), 'rad'); show('6.1 FE   v_B', at(xs, v, L) * 1e3, 'mm')
show('6.1 FE   v(1.5)', at(xs, v, 1.5) * 1e3, 'mm')
# double I
show('6.1 v_B with 2I', v_B / 2 * 1e3, 'mm'); show('6.1 v_B with 2L', -P * (2*L)**3 / (3*EI) * 1e3, 'mm')

# ---------------------------------------------------------------- Example 6.2 · simple beam, uniform load
L2, w, I2 = 8.0, 6.0, 120e-6
EI2 = E * I2                                # 24 000
show('6.2 EI', EI2, 'kN m2')
C1 = -w * L2**3 / 24                        # EI v' at x=0 from EI v = w L x^3/12 - w x^4/24 + C1 x
vfun = lambda x: (w * L2 * x**3 / 12 - w * x**4 / 24 + C1 * x) / EI2
thfun = lambda x: (w * L2 * x**2 / 4 - w * x**3 / 6 + C1) / EI2
show('6.2 hand C1 = -wL^3/24', C1, 'kN m2')
show('6.2 hand theta_A', thfun(0), 'rad'); show('6.2 hand theta_B', thfun(L2), 'rad')
show('6.2 hand v(4) = -5wL^4/384EI', vfun(4) * 1e3, 'mm'); show('6.2 hand 5wL^4/384EI', 5*w*L2**4/(384*EI2)*1e3, 'mm')
show('6.2 hand v(2)', vfun(2) * 1e3, 'mm')
xs, v, th, R = beam_fe(L2, EI2, [(0, 'pin'), (L2, 'roller')], udl=[(0, L2, -w)])
show('6.2 FE   theta_A', at(xs, th, 0), 'rad'); show('6.2 FE   v(4)', at(xs, v, 4) * 1e3, 'mm')
show('6.2 FE   v(2)', at(xs, v, 2) * 1e3, 'mm')
# span limit L/360
show('6.2 L/360', L2 / 360 * 1e3, 'mm')

# ---------------------------------------------------------------- Example 6.3 · Lab 3 beam by virtual work
L3, I3 = 6.0, 100e-6; EI3 = E * I3          # 20 000
Ay = 34.5
def M3(x):
    return np.where(x <= 3, Ay * x - 6 * x**2, np.where(x <= 4.5, Ay * x - 36 * (x - 1.5), 31.5 * (6 - x)))
m3 = lambda x: np.where(x <= 3, 0.5 * x, 0.5 * (6 - x))
parts = [simpson(lambda x: M3(x) * m3(x), a, b) for a, b in ((0, 3), (3, 4.5), (4.5, 6))]
for k, p in enumerate(parts): show(f'6.3 integral part {k+1}', p, 'kN m3')
show('6.3 sum', sum(parts), 'kN m3'); show('6.3 hand v(3) = -sum/EI', -sum(parts) / EI3 * 1e3, 'mm')
xs, v, th, R = beam_fe(L3, EI3, [(0, 'pin'), (6, 'roller')], points=[(4.5, -30)], udl=[(0, 3, -12)])
show('6.3 FE   v(3)', at(xs, v, 3) * 1e3, 'mm')
i = int(np.argmin(v)); show('6.3 FE   v_min', v[i] * 1e3, 'mm'); show('6.3 FE   x at v_min', xs[i], 'm')
show('6.3 FE   v(4.5)', at(xs, v, 4.5) * 1e3, 'mm')
show('6.3 FE   theta_A', at(xs, th, 0), 'rad'); show('6.3 FE   theta_B', at(xs, th, 6), 'rad')
# rotation at A by virtual work: unit couple at A (cw) -> m = 1 - x/6
thA = simpson(lambda x: M3(x) * (1 - x / 6), 0, 6) / EI3
show('6.3 VW theta_A (cw +)', thA, 'rad')
# fine grid for the max
xf = np.linspace(2.5, 3.5, 100001)

# ---------------------------------------------------------------- Example 6.4 · Lab 4 triangle by virtual work
A4 = 400e-6; AE = E * A4                    # 80 000 kN
nodes = {'A': (0, 0), 'B': (0, 3), 'C': (4, 0)}
members = ['AB', 'BC', 'AC']
Nreal = {'AB': 4.5, 'BC': -7.5, 'AC': 6.0}
Lm = {m: float(np.hypot(*np.subtract(nodes[m[1]], nodes[m[0]]))) for m in members}
n_h = {m: Nreal[m] / 6 for m in members}    # unit horizontal load at B
terms = {m: n_h[m] * Nreal[m] * Lm[m] for m in members}
for m in members: show(f'6.4 n N L  {m}', terms[m], 'kN m')
show('6.4 sum nNL', sum(terms.values()), 'kN m'); show('6.4 hand Delta_Bx', sum(terms.values()) / AE * 1e3, 'mm')
# vertical at B: unit load down at B. Reactions: Cy*4 = 0 (moment about A: load passes through A line x=0) -> Cy=0, Ay=1 up; AB carries -1 (compression), BC 0, AC 0
n_v = {'AB': -1.0, 'BC': 0.0, 'AC': 0.0}
show('6.4 hand Delta_By (down +)', sum(n_v[m] * Nreal[m] * Lm[m] for m in members) / AE * 1e3, 'mm')

def truss(nodes, members, sup, loads, AE):
    names = list(nodes); idx = {k: i for i, k in enumerate(names)}; n = len(names)
    K = np.zeros((2*n, 2*n)); F = np.zeros(2*n)
    for m in members:
        a, b = m[0], m[1]; (x1, y1), (x2, y2) = nodes[a], nodes[b]; L = np.hypot(x2-x1, y2-y1)
        c, s = (x2-x1)/L, (y2-y1)/L; k = AE/L*np.outer([c, s, -c, -s], [c, s, -c, -s])
        d = [2*idx[a], 2*idx[a]+1, 2*idx[b], 2*idx[b]+1]; K[np.ix_(d, d)] += k
    for k_, (fx, fy) in loads.items(): F[2*idx[k_]] += fx; F[2*idx[k_]+1] += fy
    fixed = [2*idx[k_] + (0 if ax == 'x' else 1) for k_, axs in sup.items() for ax in axs]
    free = [i for i in range(2*n) if i not in fixed]
    u = np.zeros(2*n); u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return {k_: (u[2*idx[k_]], u[2*idx[k_]+1]) for k_ in names}
U = truss(nodes, members, {'A': 'xy', 'C': 'y'}, {'B': (6, 0)}, AE)
for k_ in 'BC': show(f'6.4 FE   u_{k_}, v_{k_}', 0, f'mm  ({U[k_][0]*1e3:.5f}, {U[k_][1]*1e3:.5f})')

print('\n'.join(out))

# ---------------------------------------------------------------- extras for the text
out2 = []
xs, v, th, R = beam_fe(6.0, EI3, [(0, 'pin'), (6, 'roller')], points=[(4.5, -30)], udl=[(0, 3, -12)], nel=1200)
print('6.3 FE theta(3) rad', at(xs, th, 3), ' deg', np.degrees(at(xs, th, 3)))
i = int(np.argmin(v)); print('6.3 FE fine v_min', v[i]*1e3, 'at', xs[i])
for a, b in ((0, 3), (3, 4.5), (4.5, 6)):
    print('6.3 thetaA part', a, b, simpson(lambda x: M3(x) * (1 - x / 6), a, b))
print('deg conversions: 6.1', np.degrees(-0.0075), ' 6.2', np.degrees(-0.0053333333), ' 6.3A', np.degrees(-0.00514688))
# P6.1 cantilever, uniform 10 kN/m, 3 m, I = 6000 cm4
w1 = 10.0
print('P6.1 hand vB', -w1*3**4/(8*EI)*1e3, 'mm  thetaB', -w1*3**3/(6*EI), 'rad', np.degrees(-w1*27/(6*EI)), 'deg')
xs, v, th, R = beam_fe(3.0, EI, [(0, 'fixed')], udl=[(0, 3, -w1)])
print('P6.1 FE   vB', at(xs, v, 3)*1e3, 'thetaB', at(xs, th, 3))
# P6.2 Ex 6.2 midspan by virtual work
Mw = lambda x: 24*x - 3*x**2; mw = lambda x: np.where(x <= 4, 0.5*x, 0.5*(8-x))
print('P6.2 VW', simpson(lambda x: Mw(x)*mw(x), 0, 8), '/EI ->', -simpson(lambda x: Mw(x)*mw(x), 0, 8)/EI2*1e3, 'mm')
# P6.3 Lab 3 cantilever 8 m, 18 -> 6 kN/m, I = 50 000 cm4 -> EI = 100 000
EIc = E * 500e-6
q = lambda x: 18 - 1.5*x
def Mc(x):   # moment at x from the free-end side: load from x to 8
    # resultant of trapezoid from x to 8 about x
    a = q(x); b = 6.0; Lr = 8 - x
    return -(b*Lr**2/2 + (a - b)*Lr**2/6)
mc = lambda x: -(8 - x)
print('P6.3 M(0)', Mc(0.0), ' integral', simpson(lambda x: Mc(x)*mc(x), 0, 8), ' vB', -simpson(lambda x: Mc(x)*mc(x), 0, 8)/EIc*1e3, 'mm')
# trapezoid as many small udl pieces
pieces = [(a, a+0.02, -q(a+0.01)) for a in np.linspace(0, 7.98, 400)]
xs, v, th, R = beam_fe(8.0, EIc, [(0, 'fixed')], udl=pieces, nel=400)
print('P6.3 FE vB (piecewise udl)', at(xs, v, 8)*1e3)
# P6.4 horizontal displacement of C
print('P6.4 Delta_Cx', 1*6*4/AE*1e3, 'mm')
# strain energy truss
print('6.4 U = P Delta / 2 =', 0.5*6*1.0125, 'J (kN*mm)')
# exact maximum of 6.3 by fine double integration of M/EI (trapezoid, 6e5 steps)
x = np.linspace(0, 6, 600001); Mx = M3(x); dx = x[1] - x[0]
t = np.concatenate([[0], np.cumsum((Mx[1:] + Mx[:-1]) / 2 * dx)]); vv = np.concatenate([[0], np.cumsum((t[1:] + t[:-1]) / 2 * dx)])
C1 = -vv[-1] / 6; vv = vv + C1 * x
i = int(np.argmin(vv)); print('6.3 exact v_max', vv[i] / EI3 * 1e3, 'mm at x =', x[i], ' EI*theta_A =', C1)
# tables of the curve for 6.1 and 6.2
for x in (0, 0.75, 1.5, 2.25, 3):
    print(f'6.1 table x={x}: EIth={-60*x+10*x**2:.4f} th={(-60*x+10*x**2)/EI:.6f} EIv={-30*x**2+10/3*x**3:.4f} v={(-30*x**2+10/3*x**3)/EI*1e3:.4f} mm')
for x in (0, 1, 2, 3, 4):
    print(f'6.2 table x={x}: EIth={12*x**2-x**3-128:.4f} th={(12*x**2-x**3-128)/EI2:.6f} EIv={4*x**3-x**4/4-128*x:.4f} v={(4*x**3-x**4/4-128*x)/EI2*1e3:.4f} mm')
# P6.5: 20 kN at mid-span of the cantilever, tip deflection (reciprocity with 6.1's v(1.5))
xs, v, th, R = beam_fe(3.0, EI, [(0, 'fixed')], points=[(1.5, -20)])
print('P6.5 FE v_B', at(xs, v, 3)*1e3, ' v(1.5)', at(xs, v, 1.5)*1e3, ' theta_B', at(xs, th, 3))
mP = lambda x: -(3 - x); MP = lambda x: np.where(x <= 1.5, -20*(1.5 - x), 0*x)
print('P6.5 VW integral', simpson(lambda x: mP(x)*MP(x), 0, 1.5), ' v_B', -simpson(lambda x: mP(x)*MP(x), 0, 1.5)/EI*1e3)
# P6.6: theta_B of 6.3 by unit couple at B (counterclockwise): m = -x/6 ... sign: ccw couple at B on simple beam gives m = -x/6? check by FE
for a, b in ((0, 3), (3, 4.5), (4.5, 6)):
    print('P6.6 part', a, b, simpson(lambda x: M3(x) * (x / 6), a, b))
print('P6.6 sum', simpson(lambda x: M3(x) * (x / 6), 0, 6), ' theta_B =', simpson(lambda x: M3(x) * (x / 6), 0, 6)/EI3, ' deg', np.degrees(simpson(lambda x: M3(x) * (x / 6), 0, 6)/EI3))
