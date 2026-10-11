"""v3: Classic 101, shrunk to <= 3 in tall with a wider, more stable base (units: mm)."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2"))
from render import loft, notched_square, medallions, render, real_101, ring_marks, GLASS, GOLD

OUT = os.path.dirname(os.path.abspath(__file__))

PLINTH_W, PLINTH_H = 26.0, 3.0      # 52 mm (2.05 in) square footprint
PODIUM_W, PODIUM_H = 21.0, 14.0     # sloped podium
MOD_H, N_MOD, FLARE = 6.0, 8, 1.0   # 8 flared modules (ring ledges)
MOD_W_BOT, MOD_W_TOP = 8.8, 5.6


def classic_compact():
    z = PLINTH_H + PODIUM_H
    p = [(0, PLINTH_W), (PLINTH_H, PLINTH_W), (PLINTH_H, PODIUM_W),
         (z, MOD_W_BOT + FLARE + 1.4)]
    for i in range(N_MOD):
        w = MOD_W_BOT + (MOD_W_TOP - MOD_W_BOT) * i / (N_MOD - 1)
        p += [(z, w), (z + MOD_H, w + FLARE)]
        z += MOD_H
    wc = MOD_W_TOP * 0.82
    p += [(z, wc), (z + 2.0, wc + 0.25), (z + 2.0, wc * 0.8), (z + 3.5, wc * 0.8 + 0.15)]
    z += 3.5
    p += [(z, 2.7), (z + 3.5, 2.3)]          # short, thick spire (~5 mm dia)
    z += 3.5
    for a in np.linspace(-0.3, np.pi / 2, 10):  # rounded ball tip
        p.append((z + 2.3 * np.sin(a) + 0.6, 2.5 * np.cos(a) + 1e-3))
    return p


if __name__ == "__main__":
    p = classic_compact()
    H = max(z for z, _ in p)
    print(f"height {H:.1f} mm = {H / 25.4:.2f} in; base {2 * PLINTH_W:.0f} mm = {2 * PLINTH_W / 25.4:.2f} in square")
    shape = notched_square()
    zm = PLINTH_H + PODIUM_H - 5
    wface = PODIUM_W + (MOD_W_BOT + FLARE + 1.4 - PODIUM_W) * (zm - PLINTH_H) / PODIUM_H
    med = medallions(zm, wface, 2.8, 1.0)
    fig = plt.figure(figsize=(12, 6.5), dpi=130)
    a0 = fig.add_subplot(1, 3, 1); real_101(a0, "#5c8f8a")
    a1 = fig.add_subplot(1, 3, 2, projection="3d")
    render(a1, loft(p, shape), GLASS, med, GOLD, lim=27, H=88)
    a2 = fig.add_subplot(1, 3, 3); ring_marks(a2, p, shape, GLASS, zmin=PLINTH_H + PODIUM_H)
    a2.set_ylim(-2, 82); a2.set_xlim(-30, 36)
    a2.axhline(76.2, color="#888", lw=0.8, ls="--"); a2.text(-29, 77, "3 in limit", fontsize=7, color="#666")
    a2.set_title(f'{H:.0f} mm ({H / 25.4:.2f}") tall, {2 * PLINTH_W:.0f} mm ({2 * PLINTH_W / 25.4:.2f}") base',
                 fontsize=9)
    fig.suptitle("1. Classic 101 — compact", fontsize=15)
    fig.tight_layout(); fig.savefig(f"{OUT}/classic_101_compact.png", bbox_inches="tight")
