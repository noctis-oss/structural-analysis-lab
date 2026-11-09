"""Appendix B (load cases and combinations on the Lab 5 portal) and Appendix D (space tripod)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from frames import Frame

def portal(w, P):
    """Lab 5 portal: pin A (0,0), roller D (6,0); w kN/m down on BC, P kN right at B."""
    f = Frame(dict(A=(0, 0), B=(0, 4), C=(6, 4), D=(6, 0)), dict(AB=('A', 'B'), BC=('B', 'C'), DC=('D', 'C')))
    Dy = (6 * w * 3 + P * 4) / 6; Ay = 6 * w - Dy; Ax = -P
    f.pl += [('A', Ax, Ay), ('D', 0, Dy), ('B', P, 0)]; f.dl += [('BC', 0, -w)]
    ts = np.linspace(0, 6, 60001); Ms = [f.at('BC', t)[2] for t in ts]
    k = int(np.argmax(Ms))
    return dict(Ax=Ax, Ay=Ay, Dy=Dy, MB=f.at('AB', 4)[2], Mmax=Ms[k], at=ts[k], NDC=f.at('DC', 0)[0])

print('Appendix B · cases')
cases = dict(D=(4, 0), L=(6, 0), W=(0, 8))
for k, (w, P) in cases.items():
    r = portal(w, P); print(f'  {k}: ' + ', '.join(f'{a} {b:.3f}' for a, b in r.items()))
combos = {'1  1.4D': dict(D=1.4), '2  1.2D + 1.6L': dict(D=1.2, L=1.6), '4  1.2D + 1.0W + 1.0L': dict(D=1.2, W=1.0, L=1.0),
          '5  0.9D + 1.0W': dict(D=0.9, W=1.0), 'service D + L': dict(D=1, L=1)}
print('Appendix B · combinations (solved directly, factored loads)')
for name, fac in combos.items():
    w = sum(fac.get(k, 0) * cases[k][0] for k in cases); P = sum(fac.get(k, 0) * cases[k][1] for k in cases)
    r = portal(w, P); print(f'  {name}: w {w:.2f}, P {P:.1f} -> ' + ', '.join(f'{a} {b:.3f}' for a, b in r.items()))
# superposition check of Mmax: sum of case maxima vs true maximum of combination 4
mD, mL, mW = portal(4, 0)['Mmax'], portal(6, 0)['Mmax'], portal(0, 8)['Mmax']
print('  adding case maxima for comb 4:', 1.2 * mD + mL + mW, ' (wrong)')

print('Appendix D · tripod')
D = np.array([0, 0, 4.0]); S = dict(A=np.array([3, 0, 0.0]), B=np.array([-3, 0, 0.0]), C=np.array([0, 3, 0.0]))
def tripod(F):
    E = np.array([(S[k] - D) / np.linalg.norm(S[k] - D) for k in 'ABC']).T
    N = np.linalg.solve(E, -np.array(F, float))
    out = {}
    for i, k in enumerate('ABC'):
        e = E[:, i]; out[k] = (N[i], N[i] * e)   # bar force (tension +), reaction at the support
    return out
for F in [(3, 6, -24), (0, 0, -24), (0, 6, -24)]:
    r = tripod(F); print('  F', F)
    for k, (n, R) in r.items(): print(f'    {k}: N {n:8.3f}  R {np.round(R, 3)}')
    print('    sum R + F =', np.round(sum(R for _, R in r.values()) + np.array(F), 9))
print('  lengths', {k: float(np.linalg.norm(S[k] - D)) for k in S})

print('Appendix B · wind reversed (W acting left)')
for name, (w, P) in {'4  1.2D + 1.0W + 1.0L': (10.8, -8), '5  0.9D + 1.0W': (3.6, -8)}.items():
    r = portal(w, P); print(f'  {name}: ' + ', '.join(f'{a} {b:.3f}' for a, b in r.items()))

print('Appendix B practice')
print('  PB.1 D = 6 kN/m')
for name, (w, P) in {'1.4D': (8.4, 0), '1.2D+1.6L': (16.8, 0), '1.2D+W+L': (13.2, 8), '0.9D+W': (5.4, 8)}.items():
    r = portal(w, P); print(f'    {name}: Mmax {r["Mmax"]:.3f} at {r["at"]:.3f}, Ay {r["Ay"]:.3f}, Dy {r["Dy"]:.3f}')
print('  PB.2 D + L service:', portal(10, 0)['Mmax'])

print('Appendix D · PD.3 taller tripod, apex at z = 6')
D = np.array([0, 0, 6.0])
r = tripod((3, 6, -24))
for k, (n, R) in r.items(): print(f'    {k}: N {n:8.3f}  R {np.round(R, 3)}  L {np.linalg.norm(S[k]-D):.3f}')
