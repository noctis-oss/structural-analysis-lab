"""Lab 7 · statically indeterminate structures by the force method (compatibility). Every number printed in the lab.
Route 1: the hand method of the lab (primary structure + redundant, flexibility by virtual work, Simpson integrals).
Route 2: direct stiffness (beam elements; plane frame elements with axial stiffness; pin-jointed truss), what
Noctis Effect's Linear elastic mode does. Units kN, m. Signs: up/right +, ccw +, sagging M +."""
import numpy as np
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from lab6 import simpson, beam_fe, at
from frames import Frame

out = []
def show(k, v, u=''): print(f'{k:60s} {v:.6g} {u}')

E = 200e6
# ------------------------------------------------ Example 7.1 · propped cantilever (P6.1 cantilever + roller at B)
L, w, I = 3.0, 10.0, 60e-6; EI = E * I
D_B0 = -w * L**4 / (8 * EI)                 # P6.1, primary (cantilever) tip deflection
f_BB = L**3 / (3 * EI)                       # tip deflection per unit upward load (Ex 6.1 formula)
By = -D_B0 / f_BB
show('7.1 Delta_B0 (mm)', D_B0 * 1e3); show('7.1 f_BB (mm/kN)', f_BB * 1e3); show('7.1 By', By, 'kN')
Ay = w * L - By; MA = By * L - w * L**2 / 2
show('7.1 Ay', Ay, 'kN'); show('7.1 M_A (sagging +)', MA, 'kN m')
xm = Ay / w; Mmax = MA + Ay * xm - w * xm**2 / 2
show('7.1 x(V=0)', xm, 'm'); show('7.1 Mmax', Mmax, 'kN m')
xs, v, th, R = beam_fe(L, EI, [(0, 'fixed'), (L, 'roller')], udl=[(0, L, -w)], nel=600)
show('7.1 FE By', R[2 * (len(xs) - 1)]); show('7.1 FE Ay', R[0]); show('7.1 FE MA (ccw reaction)', R[1])
i = int(np.argmin(v)); show('7.1 FE ymin (mm)', v[i] * 1e3); show('7.1 FE x at ymin', xs[i]); show('7.1 FE theta_B (rad)', at(xs, th, L))
show('7.1 zero moment x', (Ay + np.sqrt(Ay**2 + 2 * w * MA)) / w if False else (Ay - np.sqrt(Ay**2 + 2 * w * MA)) / w)

# ------------------------------------------------ Example 7.2 · two-span beam (Ex 6.2 + roller at mid-span)
L2, w2, I2 = 8.0, 6.0, 120e-6; EI2 = E * I2
D0 = -5 * w2 * L2**4 / (384 * EI2)
f = L2**3 / (48 * EI2)
# f by virtual work: m = x/2 on each half, integral of m^2 = 2 * int_0^4 (x/2)^2 dx
fvw = 2 * simpson(lambda x: (x / 2)**2, 0, 4) / EI2
By2 = -D0 / f
show('7.2 Delta_B0 (mm)', D0 * 1e3); show('7.2 f_BB (mm/kN)', f * 1e3); show('7.2 f_BB by VW (mm/kN)', fvw * 1e3)
show('7.2 int m^2', 2 * simpson(lambda x: (x / 2)**2, 0, 4)); show('7.2 By', By2, 'kN')
Ay2 = (w2 * L2 - By2) / 2; show('7.2 Ay=Cy', Ay2)
MB = Ay2 * 4 - w2 * 16 / 2; show('7.2 M_B', MB); xm2 = Ay2 / w2; show('7.2 x(V=0)', xm2); show('7.2 Mmax', Ay2 * xm2 - w2 * xm2**2 / 2)
xs, v, th, R = beam_fe(L2, EI2, [(0, 'pin'), (4, 'roller'), (8, 'roller')], udl=[(0, 8, -w2)], nel=800)
iB = int(np.argmin(abs(xs - 4)))
show('7.2 FE Ay', R[0]); show('7.2 FE By', R[2 * iB]); show('7.2 FE Cy', R[2 * (len(xs) - 1)])
i = int(np.argmin(v[:iB])); show('7.2 FE ymin (mm)', v[i] * 1e3); show('7.2 FE x ymin', xs[i]); show('7.2 FE theta_A', th[0])
# moment where V=0 and zero-moment point
show('7.2 M zero at x', 2 * Ay2 / w2)
# settlement variant: B settles 5 mm
By_s = (-D0 - 0.005) / f; show('P7.x By with 5 mm settlement', By_s); show('   Ay', (48 - By_s) / 2, ''); show('   MB', (48 - By_s) / 2 * 4 - 48)
xs, v, th, R = beam_fe(L2, EI2, [(0, 'pin'), (8, 'roller')], points=[(4, By_s)], udl=[(0, 8, -w2)])
show('   check y(4) with that By (mm)', at(xs, v, 4) * 1e3)

# ------------------------------------------------ Example 7.3 · the Lab 5 portal with a pin at D
I3 = 10000e-8; A3 = 76e-4; EI3 = E * I3
nodes = dict(A=(0, 0), B=(0, 4), C=(6, 4), D=(6, 0)); mem = dict(AB=('A', 'B'), BC=('B', 'C'), DC=('D', 'C'))
P = Frame(nodes, mem); Dy = (60 * 3 + 8 * 4) / 6; Ay3 = 60 - Dy
P.pl += [('A', -8, Ay3), ('D', 0, Dy), ('B', 8, 0)]; P.dl += [('BC', 0, -10)]
U = Frame(nodes, mem); U.pl += [('A', -1, 0), ('D', 1, 0)]       # unit horizontal load at D, to the right
lens = dict(AB=4, BC=6, DC=4)
Delta = 0; fDD = 0
for m_ in mem:
    Mi = lambda t, m_=m_: np.array([P.at(m_, tt)[2] for tt in np.atleast_1d(t)])
    mi = lambda t, m_=m_: np.array([U.at(m_, tt)[2] for tt in np.atleast_1d(t)])
    a = simpson(lambda t: Mi(t) * mi(t), 0, lens[m_], n=200); b = simpson(lambda t: mi(t)**2, 0, lens[m_], n=200)
    show(f'7.3 {m_}: int mM', a); show(f'7.3 {m_}: int m^2', b)
    show(f'7.3 {m_}: m at 0, end', U.at(m_, 0)[2]); print('      m end', U.at(m_, lens[m_])[2])
    Delta += a; fDD += b
show('7.3 sum int mM (kN m^3)', Delta); show('7.3 sum int m^2 (m^3)', fDD)
Dx = -Delta / fDD; show('7.3 Dx (rigid axial)', Dx, 'kN'); show('7.3 Ax', -8 - Dx)
# final moments
Fm = Frame(nodes, mem); Fm.pl += [('A', -8 - Dx, Ay3), ('D', Dx, Dy), ('B', 8, 0)]; Fm.dl += [('BC', 0, -10)]
for m_, ts in (('AB', [0, 4]), ('BC', [0, 6]), ('DC', [0, 4])):
    for t in ts: print('   7.3 final', m_, t, Fm.at(m_, t))
# max in beam
tt = np.linspace(0, 6, 60001); Mb = np.array([Fm.at('BC', t)[2] for t in tt[::100]])
k = int(np.argmax(Mb)); show('7.3 beam Mmax', Mb[k]); show('   at t', tt[::100][k])
Vb = Fm.at('BC', 0)[1]; show('7.3 beam V at B', Vb); tz = Vb / 10; show('7.3 exact t(V=0)', tz); show('7.3 exact Mmax', Fm.at('BC', tz)[2])

def frame_fe(nodes, members, sup, loads, mloads, E, A, I):
    names = list(nodes); idx = {k: i for i, k in enumerate(names)}; n = len(names)
    K = np.zeros((3 * n, 3 * n)); F = np.zeros(3 * n); data = {}
    for m_, (a, b) in members.items():
        (x1, y1), (x2, y2) = nodes[a], nodes[b]; L_ = np.hypot(x2 - x1, y2 - y1); c, s_ = (x2 - x1) / L_, (y2 - y1) / L_
        k = np.zeros((6, 6)); EA, EI_ = E * A / L_, E * I
        k[np.ix_([0, 3], [0, 3])] = EA * np.array([[1, -1], [-1, 1]])
        kb = EI_ / L_**3 * np.array([[12, 6 * L_, -12, 6 * L_], [6 * L_, 4 * L_**2, -6 * L_, 2 * L_**2], [-12, -6 * L_, 12, -6 * L_], [6 * L_, 2 * L_**2, -6 * L_, 4 * L_**2]])
        k[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = kb
        T = np.zeros((6, 6)); r = np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1]]); T[:3, :3] = r; T[3:, 3:] = r
        d = [3 * idx[a], 3 * idx[a] + 1, 3 * idx[a] + 2, 3 * idx[b], 3 * idx[b] + 1, 3 * idx[b] + 2]
        K[np.ix_(d, d)] += T.T @ k @ T
        fl = np.zeros(6)
        for mm, (qx, qy) in mloads.items():          # uniform global load per unit length over the whole member
            if mm == m_:
                ql = r @ np.array([qx, qy, 0])         # local components
                fl += np.array([ql[0] * L_ / 2, ql[1] * L_ / 2, ql[1] * L_**2 / 12, ql[0] * L_ / 2, ql[1] * L_ / 2, -ql[1] * L_**2 / 12])
        F[d] += T.T @ fl; data[m_] = (d, T, k, fl)
    for nd, (fx, fy, mz) in loads.items(): F[3 * idx[nd]:3 * idx[nd] + 3] += (fx, fy, mz)
    fixed = [3 * idx[nd] + {'x': 0, 'y': 1, 'r': 2}[c] for nd, cs in sup.items() for c in cs]
    free = [i for i in range(3 * n) if i not in fixed]
    u = np.zeros(3 * n); u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    R = K @ u - F
    return {nd: (u[3 * idx[nd]], u[3 * idx[nd] + 1], u[3 * idx[nd] + 2]) for nd in names}, {nd: R[3 * idx[nd]:3 * idx[nd] + 3] for nd in names}, data, u

for label, Aval in (('realistic A=76 cm2', A3), ('rigid axial A=10 m2', 10.0)):
    Uu, RR, data, u = frame_fe(nodes, mem, dict(A='xy', D='xy'), dict(B=(8, 0, 0)), dict(BC=(0, -10)), E, Aval, I3)
    show(f'7.3 FE ({label}) Ax', RR['A'][0]); show('     Ay', RR['A'][1]); show('     Dx', RR['D'][0]); show('     Dy', RR['D'][1])
    show('     ux B (mm)', Uu['B'][0] * 1e3); show('     uy mid-beam? (n/a)', 0)
# primary structure displacement of D (roller) by FE, for the check of Delta
Uu, RR, data, u = frame_fe(nodes, mem, dict(A='xy', D='y'), dict(B=(8, 0, 0)), dict(BC=(0, -10)), E, 10.0, I3)
show('7.3 FE primary ux_D (mm)', Uu['D'][0] * 1e3); show('    hand Delta/EI (mm)', Delta / EI3 * 1e3)
Uu, RR, data, u = frame_fe(nodes, mem, dict(A='xy', D='y'), dict(D=(1, 0, 0)), {}, E, 10.0, I3)
show('7.3 FE f_DD (mm/kN)', Uu['D'][0] * 1e3); show('    hand fDD/EI (mm/kN)', fDD / EI3 * 1e3)

# ------------------------------------------------ practice
# P7.1 propped cantilever, 20 kN at mid-span (P6.5 primary)
D0p = -4.6875e-3; Byp = -D0p / f_BB; show('P7.1 By', Byp); show('P7.1 Ay', 20 - Byp); show('P7.1 MA (sagging)', Byp * 3 - 20 * 1.5); show('P7.1 M under load', Byp * 1.5)
xs, v, th, R = beam_fe(3.0, EI, [(0, 'fixed'), (3, 'roller')], points=[(1.5, -20)])
show('P7.1 FE By', R[2 * (len(xs) - 1)]); show('P7.1 FE MA ccw', R[1])
# P7.2 settlement: above (By 18.75, Ay 14.625, MB 10.5)
xs, v, th, R = beam_fe(8.0, EI2, [(0, 'pin'), (8, 'roller')], udl=[(0, 8, -6)], points=[(4, 18.75)])
show('P7.2 FE check: y(4) mm', at(xs, v, 4) * 1e3)
# P7.3 the portal without the 8 kN
P3 = Frame(nodes, mem); P3.pl += [('A', 0, 30), ('D', 0, 30)]; P3.dl += [('BC', 0, -10)]
D3 = sum(simpson(lambda t, m_=m_: np.array([P3.at(m_, tt)[2] for tt in np.atleast_1d(t)]) * np.array([U.at(m_, tt)[2] for tt in np.atleast_1d(t)]), 0, lens[m_], n=200) for m_ in mem)
Dx3 = -D3 / fDD; show('P7.3 sum mM', D3); show('P7.3 Dx', Dx3); show('P7.3 Ax', -Dx3)
F3 = Frame(nodes, mem); F3.pl += [('A', -Dx3, 30), ('D', Dx3, 30)]; F3.dl += [('BC', 0, -10)]
show('P7.3 M at B (beam)', F3.at('BC', 0)[2]); show('P7.3 M mid-beam', F3.at('BC', 3)[2])
Uu, RR, data, u = frame_fe(nodes, mem, dict(A='xy', D='xy'), {}, dict(BC=(0, -10)), E, 10.0, I3)
show('P7.3 FE rigid Dx', RR['D'][0])
Uu, RR, data, u = frame_fe(nodes, mem, dict(A='xy', D='xy'), {}, dict(BC=(0, -10)), E, A3, I3)
show('P7.3 FE A=76 Dx', RR['D'][0])
# P7.4 the bracket with a pin at C: redundant Cx; primary = Example 6.4 (Delta_Cx = 0.3 mm), f_CC = L_AC/AE
AE = E * 400e-6; f_CC = 4 / AE; Cx = -0.3e-3 / f_CC; show('P7.4 Cx', Cx); show('P7.4 N_AC = 6 + Cx', 6 + Cx)
