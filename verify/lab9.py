"""Lab 9 · verifying and interpreting results. Numbers printed in the lab (checks of app output, estimates, model variants)."""
import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import lab7
    from lab6 import beam_fe, at
E = 200e6
print('--- Ex 9.1 · audit of Example 7.3 (app values, A = 76 cm2)')
R1x, R1y, R4x, R4y = 1.187, 24.67, -9.187, 35.33
print('sum Fx', 8 + R1x + R4x, ' sum Fy', R1y + R4y - 60, ' sum M_A', R4y * 6 - 60 * 3 - 8 * 4)
U, R, d, u = lab7.frame_fe(lab7.nodes, lab7.mem, dict(A='xy', D='xy'), dict(B=(8, 0, 0)), dict(BC=(0, -10)), E, 76e-4, 10000e-8)
print('ux B, C (mm)', U['B'][0]*1e3, U['C'][0]*1e3, ' uy B, C', U['B'][1]*1e3, U['C'][1]*1e3, ' theta A, D', U['A'][2], U['D'][2])
# bounds for the beam moment (no wind part): fixed-fixed and simple
print('bounds wL2/8, wL2/12, wL2/24', 10*36/8, 10*36/12, 10*36/24)
# sway estimate: 8 kN on portal with pins, hand? skip
print('--- Ex 9.2 · I typed 120 cm4 instead of 12 000')
for I in (120e-6, 1.2e-6):
    EI = E * I; print(f'I = {I*1e8:.0f} cm4: vmax = 5wL4/384EI =', 5*6*8**4/(384*EI)*1e3, 'mm;  reactions', 24, 24, ' sigma? n/a')
print('L/360 (mm)', 8000/360)
print('--- Ex 9.3 · fixed bases for the portal')
U, R, d, u = lab7.frame_fe(lab7.nodes, lab7.mem, dict(A='xyr', D='xyr'), dict(B=(8, 0, 0)), dict(BC=(0, -10)), E, 76e-4, 10000e-8)
for k in 'AD': print(k, 'R (Fx, Fy, M)', R[k])
print('ux B mm', U['B'][0]*1e3)
# member end moments: recover via frames.Frame with reactions
from frames import Frame
F = Frame(lab7.nodes, lab7.mem); F.pl += [('A', R['A'][0], R['A'][1]), ('D', R['D'][0], R['D'][1]), ('B', 8, 0)]; F.cp += [('A', R['A'][2]), ('D', R['D'][2])]; F.dl += [('BC', 0, -10)]
for m, ts in (('AB', [0, 4]), ('BC', [0, 6]), ('DC', [0, 4])):
    for t in ts: print('  ', m, t, F.at(m, t))
tz = F.at('BC', 0)[1] / 10; print('  beam Mmax at', tz, F.at('BC', tz)[2])
U2, R2, d2, u2 = lab7.frame_fe(lab7.nodes, lab7.mem, dict(A='xy', D='xy'), dict(B=(8, 0, 0)), dict(BC=(0, -10)), E, 76e-4, 10000e-8)
print('pinned ux B mm', U2['B'][0]*1e3)
print('--- Ex 9.3b · buckling of B2 in the bracket')
# bracket bar BC: steel tube 60 x 2.2 mm (A = 4.0 cm2, I = 16.7 cm4), pinned, L = 5 m, N = -7.5 kN
D_o, t = 0.060, 0.0022; A_t = np.pi*(D_o - t)*t; I = np.pi/64*(D_o**4 - (D_o - 2*t)**4); Lb = 5.0; Fy = 250e3
Pcr = np.pi**2 * E * I / Lb**2; sig = 7.5 / 4e-4
print('tube A cm2', A_t*1e4, ' I cm4', I*1e8, ' L/r', Lb/np.sqrt(I/A_t), ' Euler limit 4.71*sqrt(E/Fy)', 4.71*np.sqrt(E/Fy))
print('Pcr', Pcr, ' FS = Pcr/7.5', Pcr / 7.5, ' sigma MPa', sig / 1e3, ' FS_y', Fy / sig)
print('--- P9.x · aluminium (E = 70 GPa) instead of steel')
print('Ex 6.2 vmax with E=70', 5*6*8**4/(384*70e6*120e-6)*1e3)
