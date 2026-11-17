"""Figure 1, TEASER: the paper's three findings, left to right, in one restrained visual language.
  a  where responses are read in frozen pi0.5, and how much of an unseen response 16 shared directions retain
  b  restoring those directions moves the commands back toward the clean ones (and what happens to success)
  c  sensitivity is not fragility: pi0 under Sensor Noise moves less than a clean rerun, yet loses success
Palette: neutral greys for the model and for references, one blue for the shared directions and their restoration,
one red for lost success. Results are horizontal bars with one large number each. No frames around panels.
Canvas is 495 x 210 pt, the same as the earlier versions, so the page layout does not change."""
import sys, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
from matplotlib.patches import Circle, Rectangle
from figstyle import *

apply()
HF = 196; DY = 14.0
f = Fig(HF + DY); f.cv.set_ylim(HF, -DY)
TT = 7.0 - DY
R = f.fig.canvas.get_renderer()
K = '#222222'
SHELL, SHELL_E = '#f3f4f6', '#9aa3ab'          # model containers
BLK = '#fbe3cb'                                 # action-expert blocks and action tokens (the one warm tone)
CARD = '#f3f5f8'                                # result panels: a tint, no outline
TRACK = '#dde2e7'; REFBAR = '#aab4bd'; DARK = '#3b4a59'
FS_B = 7.6; FS_S = 6.8; FS_HERO = 10.5


def T(x, y, s, size=FS_B, color=K, ha='left', weight='normal', **k):
    t = f.text(x, y, s, size=size, color=color, ha=ha, weight=weight, **k)
    return t.get_window_extent(R).x1 * 72 / f.fig.dpi


def arr(pts, color=K, lw=0.9, ls='-', hs=4.2, z=3.4, r=4.0):
    f.arrow(pts, color=color, lw=lw, ls=ls, hs=hs, z=z, r=r)


def tokens(xc, y, n, s=7.0, g=1.8):
    w = n * s + (n - 1) * g; x0 = xc - w / 2
    for i in range(n):
        f.box(x0 + i * (s + g), y, s, s, fc=BLK, ec=K, lw=0.55, r=1.6, z=3)
    return x0, x0 + w


def frame(x, y, s, path):
    f.cv.imshow(mi.imread(path), extent=(x, x + s, y + s, y), zorder=2, aspect='auto', interpolation='lanczos')
    f.cv.set_xlim(0, W_PT); f.cv.set_ylim(HF, -DY)
    f.cv.add_patch(Rectangle((x, y), s, s, fc='none', ec=K, lw=0.6, zorder=2.5))


def title(x, letter, text, limit):
    T(x, TT, letter, size=9.5, weight='bold', color=INK)
    xe = T(x + 11, TT + 0.3, text, size=8.2, weight='bold', color=INK)
    print(f'{letter} title ends at {xe:.1f} (limit {limit})'); return xe


def hbar(x0, y, w, color, h=6.0, z=3):
    if w > 0: f.cv.add_patch(Rectangle((x0, y - h / 2), w, h, fc=color, ec='none', zorder=z))


def whisker(xa, xb, y, color=K, lw=0.8, cap=2.0):
    f.line([(xa, y), (xb, y)], color=color, lw=lw, z=3.8)
    for xx in (xa, xb): f.line([(xx, y - cap), (xx, y + cap)], color=color, lw=lw, z=3.8)


# =================================================================== a: read and restore
xe = title(1, 'a', 'Perturbations share response directions', 246)
xe = T(xe + 3.5, TT + 0.3, 'in frozen', size=8.2, color=INK); T(xe + 2.6, TT + 0.3, r'$\pi_{0.5}$', size=8.8, color=INK)
FSZ = 46
T(1, 8.5, 'clean', size=FS_S, color=SUB); frame(1, 13, FSZ, 'extract/frame_0.png')
T(1, 71.5, 'perturbed', size=FS_S, color=SUB); frame(1, 76, FSZ, 'extract/frame_1.png')
T(1, 137.0, 'instruction', size=FS_S, color=SUB)
CT, CB = 22.0, 145.0
VX, VW = 58.0, 22.0
AX, AW = 94.0, 54.0
YS = {'L18': 38.0, 'L9': 85.0, 'L1': 119.0}; BH = 16.0
XA = AX + AW / 2
f.box(VX, CT, VW, CB - CT, fc=SHELL, ec=SHELL_E, lw=0.7, r=5, z=1.8)
T(VX + VW / 2 + 0.3, (CT + CB) / 2, 'VLM prefix', ha='center', rotation=90, rotation_mode='anchor', va='center')
arr([(1 + FSZ, 36), (VX, 36)]); arr([(1 + FSZ, 99), (VX, 99)]); arr([(37, 137), (VX, 137)])
f.box(AX, CT, AW, CB - CT, fc=SHELL, ec=SHELL_E, lw=0.7, r=5, z=1.5)
T(XA, CT + 7.0, 'Action expert', ha='center', size=FS_S, color=SUB)
for L, y in YS.items():
    f.box(AX + 8, y, AW - 16, BH, fc=BLK, ec=K, lw=0.7, r=3, z=3)
    T(XA, y + BH / 2 + 0.2, L, ha='center')
    arr([(VX + VW, y + BH / 2), (AX + 8, y + BH / 2)], lw=0.8)


def skip(xc, y_from, y_to):
    f.line([(xc, y_from), (xc, y_to + 6.0)], color=K, lw=0.9, ls=(0, (1.6, 1.5)), z=3.3)
    arr([(xc, y_to + 6.2), (xc, y_to)])


NY = 69.5
skip(XA, YS['L1'], YS['L9'] + BH); skip(XA, NY - 4.4, YS['L18'] + BH)
arr([(XA, YS['L9']), (XA, NY + 4.5)], hs=3.8)
a0, a1 = tokens(XA, 158.0, 5)
arr([(XA, 156.5), (XA, CB)])
T(a0 - 4, 161.7, 'noisy chunk', ha='right', size=FS_S, color=SUB)
tokens(XA, 3.5, 5)
arr([(XA, CT), (XA, 11.8)])
T(a0 - 4, 7.2, 'action chunk', ha='right', size=FS_S, color=SUB)

# how much of an unseen response the 16 shared directions retain, one track per recorded layer (data of Fig. 3a)
E = list(csv.DictReader(open('data/fig3_summary.csv')))
G0, G1 = 178.0, 240.0
sx = lambda v: G0 + v / 100 * (G1 - G0)
T((G0 + G1) / 2, 13.0, '16 shared directions', size=7.4, ha='center', weight='bold', color=ACC)
T((G0 + G1) / 2, 22.0, 'retain this share of it', size=FS_S, ha='center', color=SUB)
for L, y in YS.items():
    yc = y + BH / 2
    arr([(AX + AW - 8, yc), (G0 - 3, yc)], color=ACC, lw=1.1, hs=4.4, z=3.6)
    T((AX + AW + G0 - 3) / 2 + 0.5, yc - 6.6, r'$\Delta H_{%s}$' % L[1:], size=7.4, ha='center')
    vals = [float(r['energy_mean']) for r in E if r['layer'] == L]
    ctrl = max(max(float(r['energy_clean']), float(r['energy_random'])) for r in E if r['layer'] == L)
    f.line([(sx(0), yc), (sx(100), yc)], color=TRACK, lw=3.0, z=3.0)
    f.line([(sx(0), yc), (sx(ctrl), yc)], color=REFBAR, lw=3.0, z=3.1)
    for v in vals:
        f.cv.add_patch(Circle((sx(v), yc), 2.4, fc=ACC, ec='#ffffff', lw=0.5, zorder=4))
    print(L, 'shared', round(min(vals), 1), round(max(vals), 1), 'max control', round(ctrl, 1))
ya = CB + 4.0
for v, lab in ((0, '0'), (50, '50'), (100, '100%')):
    f.line([(sx(v), ya - 2.2), (sx(v), ya)], color=SUB, lw=0.6, z=3)
    T(sx(v), ya + 5.5, lab, size=FS_S, ha='center', color=SUB)
ly = 168.5
f.cv.add_patch(Circle((G0 + 2.5, ly), 2.4, fc=ACC, ec='none', zorder=4)); T(G0 + 7, ly + 0.2, 'families', size=FS_S, color=SUB)
f.line([(G0 + 36, ly), (G0 + 43, ly)], color=REFBAR, lw=3.0, z=3.1); T(G0 + 45.5, ly + 0.2, 'controls', size=FS_S, color=SUB)
# the restoration: clean values are copied back along the shared directions after L9
arr([(G0 - 3, NY), (XA + 4.6, NY)], color=ACC, lw=1.1, hs=4.4, z=3.6)
f.cv.add_patch(Circle((XA, NY), 4.2, fc='#ffffff', ec=ACC, lw=1.0, zorder=4))
f.line([(XA - 2.2, NY), (XA + 2.2, NY)], color=ACC, lw=1.2, z=4.2)
T(G0, NY + 0.2, 'restore clean values', size=FS_S, color=ACC)

# =================================================================== b: what restoration does
rows = {(r['family'], r['comparator']): r for r in csv.DictReader(open('data/fig4_summary_recomputed.csv'))}
ERR = {'Camera': dict(pert=0.409, rest=0.312), 'Sensor': dict(pert=0.209, rest=0.199)}       # Table 2
PX, BW = 254.0, 118.0
f.box(PX, 1.5, BW, 192.5, fc=CARD, ec='none', lw=0, r=7, z=1.2)
title(PX + 1, 'b', 'Restoring them cuts error', PX + BW)
LX = PX + 9.0; RX = PX + BW - 9.0
B0 = LX + 36.0; SC = 46.0 / 0.409                                # bar origin; points per unit of command error
T(LX, 14.5, 'Command error', size=FS_B, color=INK, weight='bold')
for k, (fam, name) in enumerate((('Camera', 'Camera'), ('Sensor', 'Sensor Noise'))):
    e = ERR[fam]; r = rows[(fam, 'Unedited')]; y = 33.0 + k * 38.0
    red, lo, hi = float(r['mean_of_dots']), float(r['ci_lo']), float(r['ci_hi'])
    T(LX, y, name, size=FS_B, color=INK)
    T(RX, y - 0.6, f'−{100 * red / e["pert"]:.0f}%', size=FS_HERO, weight='bold', color=ACC, ha='right')
    for j, (lab, v, col) in enumerate((('perturbed', e['pert'], REFBAR), ('restored', e['pert'] - red, ACC))):
        yy = y + 11.5 + j * 9.5
        T(LX, yy + 0.2, lab, size=FS_S, color=SUB)
        hbar(B0, yy, v * SC, col)
        xe_ = B0 + v * SC
        if j == 1:
            whisker(B0 + (e['pert'] - hi) * SC, B0 + (e['pert'] - lo) * SC, yy, color=K)
            xe_ = max(xe_, B0 + (e['pert'] - lo) * SC)
        T(xe_ + 3, yy + 0.2, f'{v:.3f}', size=FS_S, color=SUB)
f.line([(LX, 103.5), (RX, 103.5)], color='#d3dae1', lw=0.6, z=2.5)
t = 116.0; lo_, hi_ = -12, 14
T(LX, t, 'Task-success change', size=FS_B, color=INK, weight='bold'); 
AX0, AX1 = LX + 46.0, RX - 22.0
sx2 = lambda v: AX0 + (v - lo_) / (hi_ - lo_) * (AX1 - AX0)
ya2 = t + 46.0
f.line([(sx2(0), t + 9.0), (sx2(0), ya2)], color=SUB, lw=0.6, ls=(0, (2, 1.6)), z=2.8)
f.line([(AX0, ya2), (AX1, ya2)], color=SUB, lw=0.6, z=3)
for v, lab in ((-10, '−10'), (0, '0'), (10, '10 pp')):
    f.line([(sx2(v), ya2), (sx2(v), ya2 + 2.2)], color=SUB, lw=0.6, z=3)
    T(sx2(v) + (4.5 if v == 10 else 0), ya2 + 7.4, lab, size=FS_S, ha='center', color=SUB)
for k, (fam, name) in enumerate((('Camera', 'Camera'), ('Sensor', 'Sensor Noise'))):
    r = rows[(fam, 'Unedited')]; y = t + 17.0 + k * 13.5
    m, a, b = (float(r[c]) for c in ('succ_gain_pp', 'succ_lo', 'succ_hi'))
    T(LX, y + 0.2, name, size=FS_S, color=SUB)
    f.line([(sx2(a), y), (sx2(b), y)], color=ACC, lw=1.4, z=3.6)
    f.cv.add_patch(Circle((sx2(m), y), 2.6, fc=ACC, ec=CARD, lw=0.7, zorder=4))
    T(RX, y + 0.2, f'+{m:.2f}', size=FS_S, ha='right', color=INK)
T(LX, 183.0, 'intervals: simultaneous 95%', size=FS_S, color=SUB)

# =================================================================== c: sensitivity is not fragility
# pi0 under Sensor Noise against its clean rerun (Sec. 5.2): mean distance after five controls, and success lost
MOTION = dict(rerun=6.1, pert=3.5, ratio=0.58); LOSS = dict(rerun=0.0, pert=14.3, lo=2.7, hi=28.6)
CX = 378.0; CW = W_PT - 0.6 - CX
f.box(CX, 1.5, CW, 192.5, fc=CARD, ec='none', lw=0, r=7, z=1.2)
title(CX + 1, 'c', 'Sensitivity ≠ fragility', CX + CW)
LX = CX + 9.0; RX = CX + CW - 9.0
B0 = LX + 45.0
xe = T(LX, 14.5, r'$\pi_0$', size=8.2, color=INK); T(xe + 2, 14.5, 'under Sensor Noise', size=FS_B, color=INK, weight='bold')
H = 9.0
for k, (head, hero, hcol, vals, top, cols, unit) in enumerate((
        ('Early motion', f"{MOTION['ratio']:.2f}×", DARK, (MOTION['rerun'], MOTION['pert']), 7.4, (REFBAR, DARK), ' mm'),
        ('Success lost', f"{LOSS['pert']:.1f} pp", LOST, (LOSS['rerun'], LOSS['pert']), 34.5, (REFBAR, LOST), ''))):
    y = 37.0 + k * 58.0
    T(LX, y, head, size=FS_B, color=INK)
    T(RX, y - 0.6, hero, size=FS_HERO, weight='bold', color=hcol, ha='right')
    sc = (RX - 17 - B0) / top
    for j, (lab, v, col) in enumerate(zip(('clean rerun', 'Sensor Noise'), vals, cols)):
        yy = y + 15.0 + j * 13.0
        T(LX, yy + 0.2, lab, size=FS_S, color=SUB if j == 0 else INK)
        hbar(B0, yy, v * sc, col, h=H)
        if v == 0: f.line([(B0, yy - H / 2), (B0, yy + H / 2)], color=REFBAR, lw=1.4, z=3)
        xe_ = B0 + v * sc
        if k == 1 and j == 1:
            whisker(B0 + LOSS['lo'] * sc, B0 + LOSS['hi'] * sc, yy, color=K); xe_ = B0 + LOSS['hi'] * sc
        else:
            T(xe_ + 3, yy + 0.2, (f'{v:.1f}' if v else '0') + unit, size=FS_S, color=SUB)
T(CX + CW / 2, 160.0, 'moves less than a clean rerun,', size=FS_B, ha='center', color=INK)
T(CX + CW / 2, 170.5, 'yet loses more success', size=FS_B, ha='center', weight='bold', color=LOST)

out = sys.argv[1] if len(sys.argv) > 1 else 'wire/fig1_teaser'
f.save(out)
print('saved', out)
