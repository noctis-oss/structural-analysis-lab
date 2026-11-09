"""Independent equilibrium solver for the Lab 2 problems (statically determinate beams with hinges).
Sign: x right, y up, moments counterclockwise positive. Loads: point (x, Fx, Fy), couple (x, M ccw),
distributed (a, b, wa, wb) downward intensities (N/m or kN/m, consistent)."""
import numpy as np

def dist_resultant(a, b, wa, wb):
    L = b - a; W = (wa + wb) / 2 * L
    xc = a + L * (wa + 2 * wb) / (3 * (wa + wb)) if W else (a + b) / 2
    return W, xc

def solve(supports, points=(), couples=(), dist=(), hinges=()):
    # unknowns per support: pin -> Rx, Ry ; roller -> Ry ; fixed -> Rx, Ry, M
    unk = []
    for i, (x, t) in enumerate(supports):
        unk += [(i, 'Rx', x), (i, 'Ry', x)] if t == 'pin' else [(i, 'Ry', x)] if t == 'roller' else [(i, 'Rx', x), (i, 'Ry', x), (i, 'M', x)]
    loads = [(x, fx, fy) for x, fx, fy in points]
    for a, b, wa, wb in dist:
        # split distributed load at hinges for the hinge equations: keep as segments
        loads_d = (a, b, wa, wb)
    def eqs(region):
        """rows for sum Fx, sum Fy, sum M about xr for items with x in region (lo, hi, xr) """
        lo, hi, xr = region
        A = np.zeros((3, len(unk))); rhs = np.zeros(3)
        for j, (i, k, x) in enumerate(unk):
            if not (lo <= x <= hi): continue
            if k == 'Rx': A[0, j] = 1
            if k == 'Ry': A[1, j] = 1; A[2, j] = x - xr
            if k == 'M': A[2, j] = 1
        for x, fx, fy in loads:
            if lo <= x <= hi: rhs[0] -= fx; rhs[1] -= fy; rhs[2] -= fy * (x - xr)
        for x, m in couples:
            if lo <= x <= hi: rhs[2] -= m
        for a, b, wa, wb in dist:
            a2, b2 = max(a, lo), min(b, hi)
            if b2 <= a2: continue
            f = lambda x: wa + (wb - wa) * (x - a) / (b - a)
            W, xc = dist_resultant(a2, b2, f(a2), f(b2))
            rhs[1] += W; rhs[2] += W * (xc - xr)
        return A, rhs
    span = max(max(s[0] for s in supports), max([p[0] for p in points] + [c[0] for c in couples] + [d[1] for d in dist] + [0]))
    A, r = eqs((-1e9, 1e9, 0))
    rows, rhs = [A], [r]
    for h in hinges:  # moment about hinge of everything to the right of it = 0
        Ah, rh = eqs((h + 1e-12, 1e9, h)); rows.append(Ah[2:3]); rhs.append(rh[2:3])
    A = np.vstack(rows); r = np.concatenate(rhs)
    if A.shape[0] != A.shape[1] or abs(np.linalg.det(A)) < 1e-9:
        return None, A.shape
    sol = np.linalg.solve(A, r)
    return {f'{supports[i][0]}:{k}': round(v, 4) for (i, k, x), v in zip(unk, sol)}, A.shape

if __name__ == '__main__':
    print('Ex 2.1 cantilever 8 m, 18->6 kN/m:', solve([(0, 'fixed')], dist=[(0, 8, 18, 6)]))
    print('Ex 2.2 compound:', solve([(0, 'fixed'), (6, 'roller')], points=[(5, 0, -12)], couples=[(6, -9)], dist=[(0, 4, 5, 5)], hinges=[4]))
    print('Original H Ex2.11 check:', solve([(0, 'fixed'), (5, 'roller')], couples=[(5, -8)], dist=[(0, 3, 6, 6)], hinges=[3]))
    print('Original H Ex2.9 check:', solve([(0, 'fixed')], dist=[(0, 12, 15, 5)]))
    print('P2.1 triangular:', solve([(0, 'pin'), (6, 'roller')], dist=[(0, 6, 0, 30)]))
    print('P2.2 overhangs:', solve([(2, 'pin'), (6, 'roller')], points=[(0, 0, -5), (4, 0, -5), (8, 0, -5)]))
    import math
    print('P2.3 inclined:', solve([(0, 'pin'), (5, 'roller')], points=[(2, 10*math.cos(math.radians(-60)), 10*math.sin(math.radians(-60)))]))
