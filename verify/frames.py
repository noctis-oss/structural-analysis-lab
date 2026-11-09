"""Internal forces of statically determinate plane frames by sections (Lab 5).
Members run start -> end; local x along the member, local y 90 deg counterclockwise.
Conventions (Lab 3 rotated): N tension +; V = sum of local-y forces on the start side; M sagging + (= clockwise
moment of the start-side forces about the cut)."""
import math, numpy as np

class Frame:
    def __init__(s, nodes, members):
        s.n = nodes; s.m = members          # members: dict name -> (start, end)
        s.pl = []                           # (node, fx, fy)
        s.cp = []                           # (node, M ccw)
        s.dl = []                           # (member, qx, qy) uniform global, per unit length of member
    def L(s, m):
        a, b = s.m[m]; (x1, y1), (x2, y2) = s.n[a], s.n[b]; return math.hypot(x2 - x1, y2 - y1)
    def side(s, m):
        """nodes on the start side when member m is cut (structure must be a tree)."""
        start = s.m[m][0]; seen = {start}; stack = [start]
        while stack:
            k = stack.pop()
            for mm, (a, b) in s.m.items():
                if mm == m: continue
                for p, q in ((a, b), (b, a)):
                    if p == k and q not in seen: seen.add(q); stack.append(q)
        return seen
    def at(s, m, t):
        """N, V, M at distance t from the start of m."""
        a, b = s.m[m]; (x1, y1), (x2, y2) = s.n[a], s.n[b]; L = s.L(m)
        ux, uy = (x2 - x1) / L, (y2 - y1) / L; vx, vy = -uy, ux
        cx, cy = x1 + ux * t, y1 + uy * t
        side = s.side(m)
        F = np.zeros(2); Mc = 0.0
        def add(px, py, fx, fy):
            nonlocal F, Mc
            F += (fx, fy); Mc += (px - cx) * fy - (py - cy) * fx
        for (k, fx, fy) in s.pl:
            if k in side: add(*s.n[k], fx, fy)
        for (k, c) in s.cp:
            if k in side: Mc += c
        for (mm, qx, qy) in s.dl:
            p, q = s.m[mm]; Lm = s.L(mm)
            if mm == m:
                l = t; (px, py) = (x1 + ux * t / 2, y1 + uy * t / 2)
            elif p in side and q in side:
                l = Lm; (px, py) = ((s.n[p][0] + s.n[q][0]) / 2, (s.n[p][1] + s.n[q][1]) / 2)
            else: continue
            add(px, py, qx * l, qy * l)
        N = -(F[0] * ux + F[1] * uy)
        V = F[0] * vx + F[1] * vy
        M = -Mc
        return round(N, 4), round(V, 4), round(M, 4)

def table(f, pts):
    for m, ts in pts:
        for t in ts: print(f'  {m} t={t:5.2f}  N,V,M =', f.at(m, t))

if __name__ == '__main__':
    print('Ex 5.1 sign post')
    f = Frame(dict(A=(0, 0), B=(0, 4), C=(3, 4)), dict(AB=('A', 'B'), BC=('B', 'C')))
    f.pl += [('A', -8, 10), ('C', 0, -10)]; f.cp += [('A', 46)]; f.dl += [('AB', 2, 0)]
    table(f, [('AB', [0, 2, 4]), ('BC', [0, 1.5, 3])])
    print('Ex 5.2 portal')
    f = Frame(dict(A=(0, 0), B=(0, 4), C=(6, 4), D=(6, 0)), dict(AB=('A', 'B'), BC=('B', 'C'), DC=('D', 'C')))
    Dy = (60 * 3 + 8 * 4) / 6; Ay = 60 - Dy
    f.pl += [('A', -8, Ay), ('D', 0, Dy), ('B', 8, 0)]; f.dl += [('BC', 0, -10)]
    print('  Ay, Dy', Ay, Dy)
    xs = Ay / 10
    table(f, [('AB', [0, 4]), ('BC', [0, xs, 3, 6]), ('DC', [0, 4])])
    print('Ex 5.3 three-hinged')
    f = Frame(dict(A=(0, 0), D=(0, 4), C=(4, 4), E=(8, 4), B=(8, 0)), dict(AD=('A', 'D'), DC=('D', 'C'), CE=('C', 'E'), BE=('B', 'E')))
    f.pl += [('A', 6, 18), ('B', -6, 6)]; f.dl += [('DC', 0, -6)]
    table(f, [('AD', [0, 4]), ('DC', [0, 3, 4]), ('CE', [0, 4]), ('BE', [0, 4])])
