"""Figure 2 (time and protocol), same visual idiom as Figure 1: pastel blocks, thin dark outlines, black arrows,
blue = what this paper adds. Four small diagrams:
  a  one query: ten generation steps, then five controls and a new observation
  b  recording: clean and perturbed branches share the initial noise and then evolve separately
  c  restoration: at every generation step a clean reference is computed on the same plan sample and time
  d  execution study: five runs per setting from one matched state
All content follows Sec. 3.1-3.2 and Sec. 4.3 of the paper; nothing here is data."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
from matplotlib.patches import Circle, Rectangle, PathPatch, Polygon
from matplotlib.path import Path
from figstyle import *

apply()
HF = 203
f = Fig(HF)
R = f.fig.canvas.get_renderer()
K = '#222222'
C_VLM, C_AE, C_REF = '#f3f4f6', '#fbe3cb', '#fdf4ea'
FS_B = 7.8


def T(x, y, s, size=FS_B, color=K, ha='left', weight='normal', **k):
    t = f.text(x, y, s, size=size, color=color, ha=ha, weight=weight, **k)
    return t.get_window_extent(R).x1 * 72 / f.fig.dpi


def block(x, y, w, h, label, fc, size=FS_B, r=3.2, ec=K, ls='-'):
    f.box(x, y, w, h, fc=fc, ec=ec, lw=0.75, r=r, z=3, ls=ls)
    if label: T(x + w / 2, y + h / 2 + 0.2, label, size=size, ha='center')


def arr(pts, color=K, lw=0.9, ls='-', hs=4.4, z=3.4, r=4.0):
    f.arrow(pts, color=color, lw=lw, ls=ls, hs=hs, z=z, r=r)


def title(x, y, letter, text):
    T(x, y, letter, size=9.5, weight='bold', color=INK)
    T(x + 11, y + 0.3, text, size=8.2, weight='bold', color=INK)


def dots(x, y):
    for dx in (0, 4, 8):
        f.cv.add_patch(Circle((x + dx, y), 0.75, fc=K, ec='none', zorder=3.4))


def frame(x, y, s, path):
    f.cv.imshow(mi.imread(path), extent=(x, x + s, y + s, y), zorder=2, aspect='auto', interpolation='lanczos')
    f.cv.set_xlim(0, W_PT); f.cv.set_ylim(f.H, 0)
    f.cv.add_patch(Rectangle((x, y), s, s, fc='none', ec=K, lw=0.75, zorder=2.5))


BW_, BH_ = 30.0, 18.0          # one action-expert pass at generation step j

# =================================================================== a: one query
title(1, 7, 'a', 'One query: ten generation steps, then five controls')
yc = 50.0
frame(1, yc - 20, 40, 'extract/frame_0.png')
arr([(41, yc), (54, yc)])
block(54, yc - 10, 48, 20, 'VLM prefix', C_VLM)
arr([(102, yc), (120, yc)])
block(120, yc - 10, 58, 20, 'Action expert', C_AE)
arr([(168, yc - 10), (168, yc - 23), (130, yc - 23), (130, yc - 10)])
T(149, yc - 28.5, '×10', ha='center')
T(126, yc + 24.5, r'$x_0$', size=8.4, ha='center'); arr([(126, yc + 19.5), (126, yc + 10)])
arr([(178, yc), (190, yc)])
S_, G_ = 4.3, 0.75
for i in range(10):
    f.box(190 + i * (S_ + G_), yc - S_ / 2, S_, S_, fc=('#f3a968' if i < 5 else '#fdebd9'), ec=K, lw=0.5, r=0.9, z=3)
T(190 + 5 * (S_ + G_) - G_ / 2, yc - 9.5, 'action chunk', size=7.0, ha='center')
xm = 190 + 2.5 * (S_ + G_) - G_ / 2
arr([(xm, yc + 4.2), (xm, yc + 36), (21, yc + 36), (21, yc + 20.6)])
T(111, yc + 43, 'first 5 actions executed, then a new observation', size=7.0, ha='center')

# =================================================================== b: recording
LANE = '#f1f3f6'; TH = 21.0


def lane(x, y, w, h=25.0):
    f.box(x, y - h / 2, w, h, fc=LANE, ec='none', lw=0, r=5, z=1.2)


def thumb(x, y, path, label):
    """A run is named by the observation it sees: its frame, then the word."""
    f.cv.imshow(mi.imread(path), extent=(x, x + TH, y + TH / 2, y - TH / 2), zorder=2, aspect='auto', interpolation='lanczos')
    f.cv.set_xlim(0, W_PT); f.cv.set_ylim(f.H, 0)
    f.cv.add_patch(Rectangle((x, y - TH / 2), TH, TH, fc='none', ec=K, lw=0.7, zorder=2.5))
    T(x + TH + 3.5, y + 0.2, label)


BX = 256.0
title(BX, 7, 'b', 'Recording: same noise, two branches')
y1, y2 = 36.0, 80.0
T(BX + 3, (y1 + y2) / 2, r'$x_0$', size=8.4)
for y in (y1, y2):
    arr([(BX + 14, (y1 + y2) / 2), (BX + 21, (y1 + y2) / 2), (BX + 21, y), (BX + 31, y)])
XS = (BX + 31, BX + 75, BX + 136)
for y in (y1, y2):
    lane(BX + 26, y, XS[2] + BW_ + 17 - (BX + 26) + 44)
    for x, lab in zip(XS, (r'$j=0$', r'$j=1$', r'$j=9$')):
        block(x, y - BH_ / 2, BW_, BH_, lab, C_AE)
    arr([(XS[0] + BW_, y), (XS[1], y)])
    f.line([(XS[1] + BW_, y), (XS[1] + BW_ + 5, y)], color=K, lw=0.9, z=3.4)
    dots(XS[1] + BW_ + 8.5, y)
    arr([(XS[2] - 8, y), (XS[2], y)])
    arr([(XS[2] + BW_, y), (XS[2] + BW_ + 12, y)])
T(XS[2] + BW_ + 23, y1 + 0.2, 'clean'); T(XS[2] + BW_ + 23, y2 + 0.2, 'perturbed')
for x in XS:
    xc = x + BW_ / 2
    f.line([(xc, y1 + BH_ / 2), (xc, y2 - BH_ / 2)], color=ACC, lw=1.3, z=3.2)
    f.cv.add_patch(Circle((xc, (y1 + y2) / 2), 2.9, fc=ACC, ec='#ffffff', lw=0.6, zorder=4))
T(XS[1] + BW_ / 2 + 6, (y1 + y2) / 2 + 0.2, r'$\Delta H_{\ell,j}$', size=8.4, color=ACC)

# =================================================================== c: restoration
Y0 = 113.0
title(1, Y0, 'c', 'Restoration: clean reference on the same chunk, every step')
yr, ym = Y0 + 31, Y0 + 73
T(3, ym, r'$x_0$', size=8.4)
XC = (32.0, 88.0, 160.0)
lane(XC[0] - 16, yr, XC[2] + BW_ + 17 - (XC[0] - 16) + 44); lane(XC[0] - 16, ym, XC[2] + BW_ + 17 - (XC[0] - 16) + 44)
prev = 15.0
for i, (x, lab) in enumerate(zip(XC, (r'$j=0$', r'$j=1$', r'$j=9$'))):
    block(x, ym - BH_ / 2, BW_, BH_, lab, C_AE)
    block(x, yr - BH_ / 2, BW_, BH_, lab, C_REF)
    jx = x - 10
    if i < 2: arr([(prev, ym), (x, ym)])
    else: arr([(x - 14, ym), (x, ym)])
    f.cv.add_patch(Circle((jx, ym), 1.3, fc=K, ec='none', zorder=3.6))
    arr([(jx, ym), (jx, yr), (x, yr)])
    arr([(x + BW_ / 2, yr + BH_ / 2), (x + BW_ / 2, ym - BH_ / 2)], color=ACC, lw=1.5, hs=5.2, z=3.6)
    prev = x + BW_
f.line([(XC[1] + BW_, ym), (XC[1] + BW_ + 7, ym)], color=K, lw=0.9, z=3.4)
dots(XC[1] + BW_ + 11, ym)
arr([(XC[2] + BW_, ym), (XC[2] + BW_ + 12, ym)])
T(XC[2] + BW_ + 23, yr + 0.2, 'clean'); T(XC[2] + BW_ + 23, ym + 0.2, 'perturbed')
T(XC[1] + BW_ / 2 + 5, (yr + ym) / 2 - 3.6, 'restore', size=7.0, color=ACC, weight='bold'); T(XC[1] + BW_ / 2 + 5, (yr + ym) / 2 + 4.0, 'at L9', size=7.0, color=ACC, weight='bold')

# =================================================================== d: execution study
title(BX, Y0, 'd', 'Paired executions: five runs from one state')
x0, xe, ys = BX + 10, BX + 106, Y0 + 43
XEND = xe + 20                                                  # outcomes are read at the end of the episode
ends = {r'$T_0$': -29.0, r"$R_0,\,R_0'$": -5.0, r'$R_1$': 10.0, r'$T_1$': 29.0}
OK_ = {r'$T_0$': False, r"$R_0,\,R_0'$": True, r'$R_1$': True, r'$T_1$': True}
DASH_ = (0, (3.2, 1.8))


def curve(dy, n=40):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = np.array([x0, ys]), np.array([x0 + 0.41 * (xe - x0), ys + 0.12 * dy]), np.array([xe, ys + dy])
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def run(dy, ls, lw=1.0, col=K):
    c = curve(dy); f.cv.plot(c[:, 0], c[:, 1], color=col, lw=lw, ls=ls, zorder=3.3, solid_capstyle='butt', dash_capstyle='butt')


def mark(x, y, ok):
    if ok:
        f.line([(x - 2.6, y + 0.2), (x - 0.8, y + 2.3), (x + 2.8, y - 2.5)], color=K, lw=1.2, z=4)
    else:
        f.line([(x - 2.3, y - 2.3), (x + 2.3, y + 2.3)], color=LOST, lw=1.3, z=4); f.line([(x - 2.3, y + 2.3), (x + 2.3, y - 2.3)], color=LOST, lw=1.3, z=4)


up, dn = curve(ends[r"$R_0,\,R_0'$"]), curve(ends[r'$R_1$'])
f.cv.add_patch(Polygon(np.vstack([up, dn[::-1]]), closed=True, fc='#dfe4e8', ec='none', zorder=2.2))   # what a clean rerun alone changes
for lab, dy in ends.items():
    run(dy, DASH_ if lab.startswith('$T') else '-')
    f.line([(xe + 1.5, ys + dy), (XEND - 5.5, ys + dy)], color='#9a9a9a', lw=0.6, ls=(0, (1.2, 1.8)), z=2.5)
    mark(XEND, ys + dy, OK_[lab])
    T(XEND + 7, ys + dy + 0.2, lab, size=8.2)
run(-8.4, '-', lw=0.7, col='#8a8a8a')                          # the same-noise repeat R'_0 stays close to R_0
f.cv.add_patch(Circle((x0, ys), 2.2, fc=K, ec='none', zorder=4))
ya = Y0 + 80
f.line([(x0, ya), (xe, ya)], color=K, lw=0.7, z=3)
for frac, lab, top in ((0.22, '5', -9), (0.93, '60', -31)):    # each guide spans the fan at its own step
    xx = x0 + frac * (xe - x0)
    f.line([(xx, ya), (xx, ya + 2.6)], color=K, lw=0.7, z=3)
    f.line([(xx, ys + top), (xx, ya)], color='#9a9a9a', lw=0.6, ls=(0, (1.2, 1.8)), z=2.5)
    T(xx, ya + 7.6, lab, size=7.0, ha='center')
T(x0 + 0.575 * (xe - x0), ya + 7.6, 'controls', size=7.0, ha='center', color=SUB)
T(XEND, ya + 7.6, 'outcome', size=7.0, ha='center', color=SUB)
bx = XEND + 36


def brace(ya_, yb_, lab, sub):
    f.line([(bx - 3, ya_ + 1), (bx, ya_ + 1), (bx, yb_ - 1), (bx - 3, yb_ - 1)], color=K, lw=0.8, z=3.4)
    ymid = (ya_ + yb_) / 2
    T(bx + 4, ymid - 3.5, lab, size=7.0); T(bx + 4, ymid + 4.1, sub, size=7.0, color=SUB)


brace(ys - 29, ys - 5, 'perturbation', 'same noise')
brace(ys - 5, ys + 10, 'clean rerun', 'different noise')
brace(ys + 10, ys + 29, 'perturbation', 'same noise')
lx, ly = x0 + 2, ys - 30.0
for k, (ls, lab) in enumerate((('-', 'clean'), (DASH_, 'perturbed'))):
    f.line([(lx, ly + 8 * k), (lx + 13, ly + 8 * k)], color=K, lw=1.0, ls=ls, z=3.3)
    T(lx + 16, ly + 8 * k + 0.2, lab, size=7.0, color=SUB)

out = sys.argv[1] if len(sys.argv) > 1 else 'wire/fig2_new'
f.save(out)
print('saved', out)
