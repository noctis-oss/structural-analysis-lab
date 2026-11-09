"""Lab 8 · the matrix stiffness method, step by step, with the numbers printed in the lab.
Every matrix is built by hand-coded formulas (the lab's equations) and checked against an independent
assembly (lab6.truss / lab6.beam_fe) and against the app report (annex) where it prints the same matrix.
Units kN, m, rad."""
import io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from lab6 import truss, beam_fe, at
np.set_printoptions(precision=4, suppress=True, linewidth=140)
E = 200e6

print('--- Example 8.1 · two collinear bars between pins, load 10 kN at node 2')
A = 4e-4; k1 = E * A / 2; k2 = E * A / 3
print('k1, k2', k1, k2)
K = np.array([[k1, -k1, 0], [-k1, k1 + k2, -k2], [0, -k2, k2]]); print('K\n', K)
u2 = 10 / (k1 + k2); print('K_ff', k1 + k2, ' u2 (mm)', u2 * 1e3)
u = np.array([0, u2, 0]); R = K @ u - np.array([0, 10, 0]); print('R = K u - F', R)
print('N1 = k1 (u2 - u1)', k1 * u2, ' N2 = k2 (u3 - u2)', k2 * (0 - u2))
print('strain energy U = F u / 2 (J)', 0.5 * 10 * u2 * 1e3)
print('P8.1 (A2 = 2 cm2):', end=' '); k2b = E * 2e-4 / 3; u2b = 10 / (k1 + k2b)
print('k2', k2b, 'u2 mm', u2b * 1e3, 'R1', -k1 * u2b, 'R3', -k2b * u2b, 'U', 0.5 * 10 * u2b * 1e3)

print('--- Example 8.2 · the bracket truss')
nodes = {'1': (0, 0), '2': (0, 3), '3': (4, 0)}   # app numbering: N1 = A, N2 = B, N3 = C
members = [('1', '2'), ('2', '3'), ('1', '3')]
AE = E * 4e-4
Kg = np.zeros((6, 6)); idx = {'1': 0, '2': 2, '3': 4}
for a, b in members:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]; L = np.hypot(x2 - x1, y2 - y1); c, s = (x2 - x1) / L, (y2 - y1) / L
    kk = AE / L * np.array([[c*c, c*s, -c*c, -c*s], [c*s, s*s, -c*s, -s*s], [-c*c, -c*s, c*c, c*s], [-c*s, -s*s, c*s, s*s]])
    print(f'member {a}{b}: L {L} c {c} s {s} AE/L {AE/L:.1f}\n', kk)
    d = [idx[a], idx[a] + 1, idx[b], idx[b] + 1]; Kg[np.ix_(d, d)] += kk
print('K global\n', Kg)
free = [2, 3, 4]                                 # u2, v2, u3
Kff = Kg[np.ix_(free, free)]; F = np.array([6, 0, 0]); print('K_ff\n', Kff)
uf = np.linalg.solve(Kff, F); print('u_f (mm)', uf * 1e3)
# elimination by hand, as printed in the lab
a11, a12, a13 = Kff[0]; a21, a22, a23 = Kff[1]; a31, a32, a33 = Kff[2]
print('row3: u3 = (0 - a31 u2 - a32 v2)/a33 ->', a31, a32, a33)
uall = np.zeros(6); uall[free] = uf
R = Kg @ uall - np.array([0, 0, 6, 0, 0, 0]); print('R', R)
for a, b in members:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]; L = np.hypot(x2 - x1, y2 - y1); c, s = (x2 - x1) / L, (y2 - y1) / L
    du = uall[idx[b]] - uall[idx[a]]; dv = uall[idx[b] + 1] - uall[idx[a] + 1]
    print(f'N{a}{b} = AE/L (c du + s dv) =', AE / L * (c * du + s * dv), '  elongation mm', (c * du + s * dv) * 1e3)
print('det K (singular without supports)', np.linalg.det(Kg))
# P8.3: same truss, 6 kN down at B
uf3 = np.linalg.solve(Kff, np.array([0, -6, 0])); print('P8.3 u_f (mm)', uf3 * 1e3)
uall3 = np.zeros(6); uall3[free] = uf3; print('P8.3 R', Kg @ uall3 - np.array([0, 0, 0, -6, 0, 0]))
for a, b in members:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]; L = np.hypot(x2 - x1, y2 - y1); c, s = (x2 - x1) / L, (y2 - y1) / L
    print(f'  P8.3 N{a}{b}', AE / L * (c * (uall3[idx[b]] - uall3[idx[a]]) + s * (uall3[idx[b] + 1] - uall3[idx[a] + 1])))
U3 = truss(dict(A=(0, 0), B=(0, 3), C=(4, 0)), ['AB', 'BC', 'AC'], {'A': 'xy', 'C': 'y'}, {'B': (0, -6)}, AE)
print('  P8.3 check lab6.truss B, C (mm)', [x * 1e3 for x in U3['B']], [x * 1e3 for x in U3['C']])

print('--- Example 8.3 · the two-span beam by beam elements')
EI = E * 120e-6; L = 4.0; w = 6.0
k = EI / L**3 * np.array([[12, 6*L, -12, 6*L], [6*L, 4*L*L, -6*L, 2*L*L], [-12, -6*L, 12, -6*L], [6*L, 2*L*L, -6*L, 4*L*L]])
print('EI/L', EI / L, ' 4EI/L', 4 * EI / L, ' 2EI/L', 2 * EI / L, ' 12EI/L3', 12 * EI / L**3, ' 6EI/L2', 6 * EI / L**2)
feq = np.array([-w * L / 2, -w * L**2 / 12, -w * L / 2, w * L**2 / 12]); print('f_eq per element (v, th, v, th)', feq)
Kb = np.zeros((6, 6)); Fb = np.zeros(6)
for e in range(2):
    d = [2*e, 2*e+1, 2*e+2, 2*e+3]; Kb[np.ix_(d, d)] += k; Fb[d] += feq
free = [1, 3, 5]; Kff = Kb[np.ix_(free, free)]; print('K_ff (theta1, theta2, theta3)\n', Kff, '\nF_f', Fb[free])
th = np.linalg.solve(Kff, Fb[free]); print('theta (rad)', th)
ub = np.zeros(6); ub[free] = th; Rb = Kb @ ub - Fb; print('R (v rows)', Rb[[0, 2, 4]])
for e in range(2):
    d = [2*e, 2*e+1, 2*e+2, 2*e+3]; fe = k @ ub[d] - feq
    print(f'element {e+1} end forces k d - f_eq (V0, M0, V1, M1):', fe)
xs, v, thf, R = beam_fe(8.0, EI, [(0, 'pin'), (4, 'roller'), (8, 'roller')], udl=[(0, 8, -w)])
print('check lab6.beam_fe theta(0), theta(4), theta(8)', at(xs, thf, 0), at(xs, thf, 4), at(xs, thf, 8))

print('--- P8.2 · the propped cantilever (Ex 7.1) with one element')
EIc = E * 60e-6; Lc = 3.0; wc = 10.0
thB = (wc * Lc**2 / 12) / (4 * EIc / Lc); print('theta_B', thB)
MA = 2 * EIc / Lc * thB + wc * Lc**2 / 12; print('M_A (ccw, reaction)', MA)
VA = 6 * EIc / Lc**2 * thB + wc * Lc / 2; print('A_y', VA, ' B_y', wc * Lc - VA)
