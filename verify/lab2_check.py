"""Independent check of every Lab 2 number (rigid-body equilibrium with numpy)."""
import numpy as np

def solve(unknowns, loads, extra=None):
    """unknowns: list of (name, point, kind) kind in 'x','y','m'; loads: list of (point,(fx,fy)) ; extra rows."""
    A = np.zeros((3, len(unknowns))); b = np.zeros(3)
    for j, (n, (x, y), k) in enumerate(unknowns):
        if k == 'x': A[0, j] = 1; A[2, j] = -y
        if k == 'y': A[1, j] = 1; A[2, j] = x
        if k == 'm': A[2, j] = 1
    for (x, y), (fx, fy) in loads:
        b -= [fx, fy, x * fy - y * fx]
    if extra:
        A = np.vstack([A, extra[0]]); b = np.append(b, extra[1])
    return dict(zip([u[0] for u in unknowns], np.linalg.solve(A, b)))

print('Ex 2.1', solve([('Ax', (0, 0), 'x'), ('Ay', (0, 0), 'y'), ('Dy', (9, 0), 'y')], [((3, 0), (0, -12)), ((6, 0), (0, -6)), ((3, 3), (4, 0))]))
print('Ex 2.2', solve([('Ax', (0, 0), 'x'), ('Ay', (0, 0), 'y'), ('MA', (0, 0), 'm')], [((0, 2), (8, 0)), ((3, 4), (0, -10))]))
# three-hinged: extra row = moment of B forces about C(4,4) = 0 (CB unloaded): 4*By + 4*Bx = 0
U = [('Ax', (0, 0), 'x'), ('Ay', (0, 0), 'y'), ('Bx', (8, 0), 'x'), ('By', (8, 0), 'y')]
print('Ex 2.3', solve(U, [((2, 4), (0, -24))], ([0, 0, 4, 4], 0)))
print('P2.4  ', solve(U, [((0, 4), (12, 0))], ([0, 0, 4, 4], 0)))
print('P2.2  ', solve([('Ax', (0, 0), 'x'), ('Ay', (0, 0), 'y'), ('Dy', (9, 0), 'y')], [((1.5, .866), (0, -4)), ((4.5, 2.598), (0, -6)), ((7.5, .866), (0, -4)), ((1.5, .866), (3, 0))]))
print('P2.3  ', solve([('Ax', (0, 0), 'x'), ('Ay', (0, 0), 'y'), ('Cy', (4, 3), 'y')], [((2, 3), (0, -20)), ((0, 3), (6, 0))]))
