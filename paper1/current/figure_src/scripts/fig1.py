"""Figure 1, TEASER: the paper's three findings, left to right.
  a  where responses are read in frozen pi0.5, and how much of an unseen response 16 shared directions retain
  b  restoring those directions moves the commands back toward the clean ones (and what happens to success)
  c  sensitivity is not fragility: pi0 under Sensor Noise moves less than a clean rerun, yet loses success
Blue = the shared directions and their restoration; grey = references and controls; red = lost success.
Canvas is 495 x 210 pt, the same as the earlier version, so the page layout does not change."""
import sys, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
from matplotlib.patches import Circle, Rectangle, Polygon
from figstyle import *

apply()
HF = 196; DY = 14.0
f = Fig(HF + DY); f.cv.set_ylim(HF, -DY)
TT = 7.0 - DY
R = f.fig.canvas.get_renderer()
K = '#222222'; GREY = '#f2f2f2'
C_VLM, C_AE, C_TOK = '#fdf0c0', '#fcd9b8', '#fcd9b8'
CARD_B, CARD_C, EDGE_C = '#f4f8fd', '#fdf5f2', '#d9a79b'
FS_B = 7.8


def T(x, y, s, size=FS_B, color=K, ha='left', weight='normal', **k):
    t = f.text(x, y, s, size=size, color=color, ha=ha, weight=weight, **k)
    return t.get_window_extent(R).x1 * 72 / f.fig.dpi


def block(x, y, w, h, label, fc, size=FS_B, r=3.2):
    f.box(x, y, w, h, fc=fc, ec=K, lw=0.75, r=r, z=3)
    if label: T(x + w / 2, y + h / 2 + 0.2, label, size=size, ha='center')


def arr(pts, color=K, lw=0.9, ls='-', hs=4.4, z=3.4, r=4.0):
    f.arrow(pts, color=color, lw=lw, ls=ls, hs=hs, z=z, r=r)


def tokens(xc, y, n, s=7.4, g=1.9):
    w = n * s + (n - 1) * g; x0 = xc - w / 2
    for i in range(n):
        f.box(x0 + i * (s + g), y, s, s, fc=C_TOK, ec=K, lw=0.6, r=1.7, z=3)
    return x0, x0 + w


def frame(x, y, s, path, ec=K, lw=0.75):
    f.cv.imshow(mi.imread(path), extent=(x, x + s, y + s, y), zorder=2, aspect='auto', interpolation='lanczos')
    f.cv.set_xlim(0, W_PT); f.cv.set_ylim(HF, -DY)
    f.cv.add_patch(Rectangle((x, y), s, s, fc='none', ec=ec, lw=lw, zorder=2.5))


def title(x, letter, text, limit):
    T(x, TT, letter, size=9.5, weight='bold', color=INK)
    xe = T(x + 11, TT + 0.3, text, size=8.2, weight='bold', color=INK)
    print(f'{letter} title ends at {xe:.1f} (limit {limit})'); return xe


# =================================================================== a: read and restore
xe = title(1, 'a', 'Perturbations share response directions', 244)
xe = T(xe + 3.5, TT + 0.3, 'in frozen', size=8.2, color=INK); T(xe + 2.6, TT + 0.3, r'$\pi_{0.5}$', size=8.8, color=INK)
FSZ = 42
T(1, 6.0, 'clean'); frame(1, 11, FSZ, 'extract/frame_0.png')
T(1, 67.0, 'perturbed'); frame(1, 72, FSZ, 'extract/frame_1.png')
T(1, 133.5, 'instruction')
VX, VW = 50.0, 28.0
AX, AW = 94.0, 54.0
CT, CB = 20.0, 143.0
YS = {'L18': 35.0, 'L9': 83.0, 'L1': 118.0}; BH = 17.0
f.box(VX, CT, VW, CB - CT, fc=C_VLM, ec=K, lw=0.75, r=5, z=1.8)
T(VX + VW / 2, (CT + CB) / 2 - 4.3, 'VLM', ha='center'); T(VX + VW / 2, (CT + CB) / 2 + 4.7, 'prefix', ha='center')
arr([(1 + FSZ, 32), (VX, 32)]); arr([(1 + FSZ, 93), (VX, 93)]); arr([(39, 133.5), (VX, 133.5)])
f.box(AX, CT, AW, CB - CT, fc=GREY, ec=K, lw=0.75, r=5, z=1.5)
T(AX + AW / 2, 27.5, 'Action expert', ha='center')
XA = AX + AW / 2
for L, y in YS.items():
    block(AX + 7, y, AW - 14, BH, L, C_AE)
    arr([(VX + VW, y + BH / 2), (AX + 7, y + BH / 2)])


def skip(xc, y_from, y_to):
    f.line([(xc, y_from), (xc, y_to + 6.2)], color=K, lw=0.9, ls=(0, (1.6, 1.5)), z=3.3)
    arr([(xc, y_to + 6.4), (xc, y_to)])


NY = 67.5
skip(XA, YS['L1'], YS['L9'] + BH); skip(XA, NY - 4.4, YS['L18'] + BH)
arr([(XA, YS['L9']), (XA, NY + 4.5)], hs=4.0)
a0, a1 = tokens(XA, 157.0, 5)
arr([(XA, 155.5), (XA, CB)])
T(a0 - 4, 161.0, 'noisy chunk', ha='right')
tokens(XA, 1.0, 5)
arr([(XA, CT), (XA, 9.6)])
T(a1 + 4, 5.0, 'action chunk')

# what 16 shared directions retain of an unseen response, one row per recorded layer (data of Fig. 3a)
E = list(csv.DictReader(open('data/fig3_summary.csv')))
GX, GW = 172.0, 72.0
GT_, GB_ = 15.0, 161.0
f.box(GX, GT_, GW, GB_ - GT_, fc=CARD_B, ec=ACC, lw=0.9, r=5, z=1.6)
T(GX + GW / 2, 22.6, '16 shared directions', size=6.8, ha='center', weight='bold', color=INK)
T(GX + GW / 2, 30.6, 'energy retained (%)', size=6.6, ha='center', color=SUB)
PX0, PX1 = GX + 9, GX + GW - 11
sx = lambda v: PX0 + v / 100 * (PX1 - PX0)
ya = 146.5
for v, lab in ((0, '0'), (50, '50'), (100, '100')):
    f.line([(sx(v), 36.0), (sx(v), ya)], color='#cfdcec', lw=0.5, z=2.2)
    f.line([(sx(v), ya), (sx(v), ya + 2.4)], color=K, lw=0.7, z=3)
    T(sx(v), ya + 7.4, lab, size=6.6, ha='center')
f.line([(PX0, ya), (PX1, ya)], color=K, lw=0.7, z=3)
for L, y in YS.items():
    yc = y + BH / 2
    arr([(AX + AW - 7, yc), (GX, yc)], color=ACC, lw=1.1, hs=4.6, z=3.6)
    T((AX + AW + GX) / 2 + 0.3, yc - 6.6, r'$\Delta H_{%s}$' % L[1:], size=7.6, ha='center')
    vals = [float(r['energy_mean']) for r in E if r['layer'] == L]
    ctrl = max(max(float(r['energy_clean']), float(r['energy_random'])) for r in E if r['layer'] == L)
    f.line([(sx(0), yc + 5.6), (sx(ctrl), yc + 5.6)], color=RAND, lw=2.4, z=3.2)
    f.line([(sx(0), yc), (sx(max(vals)), yc)], color=ACC_M, lw=0.9, z=3.25)
    for v in vals:
        f.cv.add_patch(Circle((sx(v), yc), 2.2, fc=ACC, ec='#ffffff', lw=0.45, zorder=4))
    print(L, 'shared', round(min(vals), 1), round(max(vals), 1), 'max control', round(ctrl, 1))
# the restoration: the part of the L9 response along the shared directions is taken out again
arr([(GX, NY), (XA + 4.6, NY)], color=ACC, lw=1.1, hs=4.6, z=3.6)
f.cv.add_patch(Circle((XA, NY), 4.4, fc='#ffffff', ec=ACC, lw=1.0, zorder=4))
f.line([(XA - 2.3, NY), (XA + 2.3, NY)], color=ACC, lw=1.2, z=4.2)
f.text(GX + 5.0, NY + 0.2, 'restore clean values', size=6.6, color=ACC, weight='bold', zorder=5, bbox=dict(fc=CARD_B, ec='none', pad=0.8))
ly = 170.5
f.cv.add_patch(Circle((GX + 5, ly), 2.2, fc=ACC, ec='none', zorder=4)); T(GX + 9.5, ly + 0.2, 'families', size=6.6, color=SUB)
f.line([(GX + 38, ly), (GX + 45, ly)], color=RAND, lw=2.4, z=3.2); T(GX + 47.5, ly + 0.2, 'controls', size=6.6, color=SUB)
T(GX + GW / 2, 180.0, 'held-out tasks, families', size=6.6, ha='center', color=SUB)

# =================================================================== b: what restoration does
rows = {(r['family'], r['comparator']): r for r in csv.DictReader(open('data/fig4_summary_recomputed.csv'))}
ERR = {'Camera': dict(pert=0.409, rest=0.312, whole=0.297), 'Sensor': dict(pert=0.209, rest=0.199, whole=0.194)}   # Table 2
PX, BW = 252.0, 126.0
f.box(PX, 1.5, BW, 192.5, fc=CARD_B, ec=ACC, lw=0.8, r=6, z=1.5)
title(PX + 1, 'b', 'Restoring them cuts error', PX + BW)
TX0 = PX + 7.0
T(TX0, 13.0, 'Command error', size=7.6, color=INK)
BX0, BX1 = PX + 42.0, PX + BW - 22.0
bx = lambda v: BX0 + v * (BX1 - BX0)
T(BX0, 24.0, 'clean', size=6.6, color=SUB, ha='center'); T(BX1, 24.0, 'perturbed', size=6.6, color=SUB, ha='center')
for k, (fam, label) in enumerate((('Camera', ('Camera',)), ('Sensor', ('Sensor', 'Noise')))):
    e = ERR[fam]; r = rows[(fam, 'Unedited')]; y = 45.0 + k * 33.0
    red = float(r['mean_of_dots']); lo, hi = float(r['ci_lo']), float(r['ci_hi'])
    xr, xw = bx(1 - red / e['pert']), bx(e['whole'] / e['pert'])
    if len(label) == 1: T(TX0, y + 0.2, label[0], size=7.0)
    else: T(TX0, y - 3.6, label[0], size=7.0); T(TX0, y + 4.0, label[1], size=7.0)
    f.line([(BX0, y), (BX1, y)], color='#b7c3cf', lw=1.0, z=3)                      # the whole distance from clean to perturbed
    f.line([(BX0, y - 3), (BX0, y + 3)], color=K, lw=0.8, z=3.2)
    f.cv.add_patch(Rectangle((xw, y - 4.2), BX1 - xw, 8.4, fc=ACC_M, ec='none', alpha=0.55, zorder=2.9))   # what whole-L9 replacement removes
    f.line([(bx(1 - hi / e['pert']), y + 6.6), (bx(1 - lo / e['pert']), y + 6.6)], color=ACC, lw=0.9, z=3.6)   # simultaneous 95% interval of the reduction
    for xx in (bx(1 - hi / e['pert']), bx(1 - lo / e['pert'])):
        f.line([(xx, y + 5.0), (xx, y + 8.2)], color=ACC, lw=0.9, z=3.6)
    if BX1 - xr > 7:
        arr([(BX1, y), (xr, y)], color=ACC, lw=2.0, hs=5.6, z=3.8)
    else:
        f.line([(BX1, y), (xr + 2, y)], color=ACC, lw=2.0, z=3.8)
        f.cv.add_patch(Polygon([(xr - 1.6, y), (xr + 3.2, y - 2.8), (xr + 3.2, y + 2.8)], closed=True, fc=ACC, ec='none', zorder=3.9))
    f.cv.add_patch(Rectangle((BX1 - 2.4, y - 2.4), 4.8, 4.8, fc=K, ec='#ffffff', lw=0.5, zorder=4))
    pct = 100 * red / e['pert']; share = 100 * (e['pert'] - e['rest']) / (e['pert'] - e['whole'])
    T(min((xr + BX1) / 2, BX1 - 4), y - 10.0, f'−{pct:.0f}%', size=9.0, weight='bold', color=ACC, ha='center')
    print(fam, 'reduction %', round(pct, 1), 'share of whole-L9', round(share, 1))
f.cv.add_patch(Rectangle((TX0, 95.4), 7, 5.2, fc=ACC_M, ec='none', alpha=0.55, zorder=3))
T(TX0 + 10, 98.2, 'reduction with whole L9 replaced', size=6.6, color=SUB)
f.line([(TX0, 107.0), (TX0 + 7, 107.0)], color=ACC, lw=0.9, z=3.6)
for xx in (TX0, TX0 + 7): f.line([(xx, 105.5), (xx, 108.5)], color=ACC, lw=0.9, z=3.6)
T(TX0 + 10, 107.2, 'simultaneous 95% interval', size=6.6, color=SUB)
f.line([(TX0, 116.5), (PX + BW - 7.0, 116.5)], color='#c9d8ea', lw=0.6, z=2.5)

AX0, AX1 = PX + 42.0, PX + 92.0
VX1 = PX + BW - 7.0
t = 127.0; lo_, hi_ = -12, 14
T(TX0, t, 'Task-success change (pp)', size=7.6, color=INK)
sx2 = lambda v: AX0 + (v - lo_) / (hi_ - lo_) * (AX1 - AX0)
ya2 = t + 45.5
f.line([(AX0, ya2), (AX1, ya2)], color=K, lw=0.7, z=3)
for v, lab in ((-10, '−10'), (0, '0'), (10, '10')):
    f.line([(sx2(v), ya2), (sx2(v), ya2 + 2.6)], color=K, lw=0.7, z=3)
    T(sx2(v), ya2 + 7.8, lab, size=7.0, ha='center')
f.line([(sx2(0), t + 8.0), (sx2(0), ya2)], color='#7a7a7a', lw=0.7, ls=(0, (2, 1.6)), z=2.8)
for k, (fam, label) in enumerate((('Camera', ('Camera',)), ('Sensor', ('Sensor', 'Noise')))):
    r = rows[(fam, 'Unedited')]; y = t + 16.5 + k * 17.5
    m, a, b = (float(r[c]) for c in ('succ_gain_pp', 'succ_lo', 'succ_hi'))
    if len(label) == 1: T(TX0, y + 0.2, label[0], size=7.0)
    else: T(TX0, y - 3.6, label[0], size=7.0); T(TX0, y + 4.0, label[1], size=7.0)
    f.line([(sx2(a), y), (sx2(b), y)], color=ACC, lw=1.4, z=3.6)
    f.cv.add_patch(Circle((sx2(m), y), 2.7, fc=ACC, ec=CARD_B, lw=0.7, zorder=4))
    T(VX1, y + 0.2, f'+{m:.2f}', size=7.0, ha='right', weight='bold')

# =================================================================== c: sensitivity is not fragility
# pi0 under Sensor Noise against its clean rerun (Sec. 5.2): mean distance after five controls, and success lost
MOTION = dict(rerun=6.1, pert=3.5, ratio=0.58); LOSS = dict(rerun=0.0, pert=14.3, lo=2.7, hi=28.6)
CX, CW = 386.0, W_PT - 0.6 - 386.0
f.box(CX, 1.5, CW, 192.5, fc=CARD_C, ec=EDGE_C, lw=0.8, r=6, z=1.5)
title(CX + 1, 'c', 'Sensitivity ≠ fragility', CX + CW)
T(CX + CW / 2, 13.0, r'$\pi_0$ under Sensor Noise', size=7.4, ha='center', color=INK)
YB, YT = 137.0, 54.0                                  # bar baseline and the top of the tallest bar
BWd = 18.0
SLATE = '#3f5568'; PALE = '#b9c3cc'


def bars(xc, title_, unit, vals, top, cols, fmt, ci=None, tag=None):
    T(xc, 27.5, title_[0], size=7.0, ha='center', color=INK); T(xc, 35.5, title_[1], size=7.0, ha='center', color=INK)
    f.line([(xc - 24, YB), (xc + 24, YB)], color=K, lw=0.7, z=3)
    for i, (v, c) in enumerate(zip(vals, cols)):
        x = xc - 11.0 + i * 22.0
        h = v / top * (YB - YT)
        if h > 0: f.cv.add_patch(Rectangle((x - BWd / 2, YB - h), BWd, h, fc=c, ec='none', zorder=3))
        if ci and i == 1:
            y1, y2 = YB - ci[0] / top * (YB - YT), YB - ci[1] / top * (YB - YT)
            f.line([(x, y1), (x, y2)], color=K, lw=0.8, z=3.6)
            for yy in (y1, y2): f.line([(x - 2.2, yy), (x + 2.2, yy)], color=K, lw=0.8, z=3.6)
        if ci and i == 1:
            T(x - BWd / 2 - 2.5, YB - h + 0.5, fmt(v), size=9.0, ha='right', color=c, weight='bold', zorder=5)
        elif h > 12:
            T(x, YB - h + 6.0, fmt(v), size=6.8, ha='center', color='#ffffff' if i == 1 else INK, weight='bold' if i == 1 else 'normal', zorder=5)
        else:
            T(x, YB - 5.5, fmt(v), size=6.8, ha='center', color=SUB)
        for j, w in enumerate(('clean', 'rerun') if i == 0 else ('Sensor', 'Noise')):
            T(x, YB + 7.0 + 7.4 * j, w, size=6.6, ha='center', color=SUB if i == 0 else K)
    if tag: T(*tag[:2], tag[2], size=8.6, ha='center', weight='bold', color=tag[3])


x1, x2 = CX + 28.5, CX + CW - 28.5
hm = MOTION['pert'] / 7.0 * (YB - YT); hl = LOSS['hi'] / 30.0 * (YB - YT)
bars(x1, ('early', 'motion (mm)'), 'mm', (MOTION['rerun'], MOTION['pert']), 7.0, (PALE, SLATE), lambda v: f'{v:.1f}',
     tag=(x1 + 13.5, YB - hm - 7.0, f"{MOTION['ratio']:.2f}×", SLATE))
bars(x2, ('success', 'lost (pp)'), 'pp', (LOSS['rerun'], LOSS['pert']), 30.0, (PALE, LOST), lambda v: f'{v:.1f}' if v else '0', ci=(LOSS['lo'], LOSS['hi']),
     tag=(x2 - 11.0, YB - LOSS['pert'] / 30.0 * (YB - YT) - 7.5, '', LOST))
T(CX + CW / 2, 173.0, 'moves less than a rerun,', size=7.2, ha='center', weight='bold', color=INK)
T(CX + CW / 2, 182.5, 'yet fails more', size=7.2, ha='center', weight='bold', color=LOST)

out = sys.argv[1] if len(sys.argv) > 1 else 'wire/fig1_teaser'
f.save(out)
print('saved', out)
