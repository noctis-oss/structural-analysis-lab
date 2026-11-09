"""Lab 9 case study: the 41-bar crane truss (Mustapha, SolidWorks 2022, Ch. 2, Fig. 2.4; Megson Ex. 4.11).
Independent direct-stiffness solution and the hand route used in the book."""
import numpy as np
N = {"A": (9, 0), "B": (12, 0), "C": (9, 3), "D": (12, 3), "E": (9, 6), "F": (12, 6), "G": (9, 9), "H": (12, 9), "I": (9, 12), "J": (12, 12),
     "K": (9, 15), "L": (12, 15), "O": (10.5, 18), "R": (0, 9), "N": (3, 9), "M": (6, 9), "Q": (3, 12), "P": (6, 12), "T": (15, 9), "W": (18, 9),
     "S": (15, 12), "V": (18, 12)}
MEM = "A-C C-E E-G G-I I-K B-D D-F F-H H-J J-L A-B C-D E-F G-H I-J K-L C-B E-D G-F I-H K-J O-K O-L Q-P P-I R-N N-M M-G R-Q Q-N P-M N-P P-G J-S S-V H-T T-W S-T V-W J-T T-V".split()
E, A = 210e6, np.pi / 4 * (0.2**2 - 0.08**2)          # kN/m², m²
keys = list(N); ix = {k: i for i, k in enumerate(keys)}; n = len(keys)
K = np.zeros((2 * n, 2 * n)); F = np.zeros(2 * n)
F[2 * ix["R"] + 1] = -1500; F[2 * ix["W"] + 1] = -2000
geo = {}
for m in MEM:
    a, b = m.split('-'); (x1, y1), (x2, y2) = N[a], N[b]; L = np.hypot(x2 - x1, y2 - y1); c, s = (x2 - x1) / L, (y2 - y1) / L
    k = E * A / L * np.outer([-c, -s, c, s], [-c, -s, c, s]); d = [2 * ix[a], 2 * ix[a] + 1, 2 * ix[b], 2 * ix[b] + 1]
    K[np.ix_(d, d)] += k; geo[m] = (a, b, c, s, L)
fixed = [2 * ix["A"], 2 * ix["A"] + 1, 2 * ix["B"] + 1]; free = [i for i in range(2 * n) if i not in fixed]
u = np.zeros(2 * n); u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
R = K @ u - F
force = {}
for m, (a, b, c, s, L) in geo.items():
    du = u[2 * ix[b]:2 * ix[b] + 2] - u[2 * ix[a]:2 * ix[a] + 2]; force[m] = E * A / L * (c * du[0] + s * du[1])
print(f"A = {A*1e4:.2f} cm²  EA = {E*A:.4e} kN")
print("reactions: Ax %.3f  Ay %.3f  By %.3f kN" % (R[2*ix['A']], R[2*ix['A']+1], R[2*ix['B']+1]))
for m in ["I-H", "P-I", "M-G", "P-G", "J-S", "H-T", "J-T", "G-I", "H-J", "A-C", "B-D", "C-B", "R-N", "R-Q"]:
    print(f"  {m:4s} {force[m]:10.3f} kN")
mx = max(keys, key=lambda k: np.hypot(*u[2*ix[k]:2*ix[k]+2]))
for k in ["R", "W", mx]:
    print(f"  node {k}: ux {u[2*ix[k]]*1000:9.4f}  uy {u[2*ix[k]+1]*1000:9.4f} mm  |d| {np.hypot(*u[2*ix[k]:2*ix[k]+2])*1000:.4f} mm")
print("largest tension %s %.1f, largest compression %s %.1f" % (max(force, key=force.get), max(force.values()), min(force, key=force.get), min(force.values())))
print("zero-force:", [m for m in force if abs(force[m]) < 1e-6])
# hand route: section through the tower between y = 9 and 12 with both arms attached
print("hand IH = (3000 + 1500 - 4000)/0.7071 =", (3000 + 1500 - 4000) / np.sqrt(0.5))
# virtual work for vR (unit load at R): sum n N L / EA

# check 5: the most compressed bar G-H (3 m) and the most stressed tension bar P-I
I = np.pi / 64 * (0.2**4 - 0.08**4)
Pcr = np.pi**2 * E * I / 3**2
print(f"I = {I*1e8:.0f} cm4, r = {np.sqrt(I/A)*100:.2f} cm, Pcr(G-H, 3 m) = {Pcr:.0f} kN, FS = {Pcr/4500:.2f}")
print(f"sigma max = {4500/A/1000:.1f} MPa; FS yield with Fy 250 MPa = {250/(4500/A/1000):.2f}, 355 MPa = {355/(4500/A/1000):.2f}")
# right-arm and left-arm sections
print("right arm: JT", 2000/np.sqrt(.5), "JS", 2000*3/3, "HT", -(2000+2000))
print("left arm: PI", 1500*9/3, "PG", -1500/np.sqrt(.5), "MG", -(4500-1500))
