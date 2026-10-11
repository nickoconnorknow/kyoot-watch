"""v2 concepts: Taipei 101 ring stands that read clearly as the tower (units: mm).

Key Taipei 101 tells kept in every concept:
  * 8 stacked modules that each flare OUTWARD going up (the "stacked boxes" look)
  * sloped pyramid podium at the bottom
  * round "ruyi" medallions on the four faces just above the podium
  * square plan with saw-tooth notched corners
  * narrow crown tiers + spire on top (kept short and thick here)
The flared modules double as ring ledges: a ring slides down until it rests on
the top lip of whichever module matches its size.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = os.path.dirname(os.path.abspath(__file__))
LIGHT = np.array([-0.55, -0.75, 0.6]); LIGHT /= np.linalg.norm(LIGHT)


# ---------- cross sections (unit half-width, closed CCW loops, same M points) ----------
def notched_square(n=0.13, m_per_edge=3):
    """Square with Taipei 101's double saw-tooth notch at each corner."""
    q = [(1, -(1 - 2 * n)), (1, 1 - 2 * n), (1 - n, 1 - 2 * n), (1 - n, 1 - n),
         (1 - 2 * n, 1 - n), (1 - 2 * n, 1)]
    pts = []
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        for x, y in q:
            pts.append((c * x - s * y, s * x + c * y))
    return np.array(pts)


def chamfered_square(n=0.18):
    q = [(1, -(1 - n)), (1, 1 - n), (1 - n, 1)]
    pts = []
    for k in range(4):
        c, s = np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)
        for x, y in q:
            pts.append((c * x - s * y, s * x + c * y))
    return np.array(pts)


def circle(m=48):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False) + np.pi / m
    return np.c_[np.cos(t), np.sin(t)]


# ---------- profiles: list of (z, half_width) keypoints; repeated z = a ledge ----------
def tower_profile(z0, podium_w0, podium_h, mod_w_bot, mod_w_top, flare, n_mod=8,
                  mod_h=10.5, spire=True):
    pw_top = mod_w_bot + flare + 1.6
    p = [(z0, podium_w0 + 4), (z0 + 3, podium_w0 + 4), (z0 + 3, podium_w0),   # plinth slab
         (z0 + podium_h, pw_top)]                                # tall sloped podium
    z = z0 + podium_h
    for i in range(n_mod):
        w = mod_w_bot + (mod_w_top - mod_w_bot) * i / (n_mod - 1)
        p += [(z, w), (z + mod_h, w + flare)]                    # flares outward going up
        z += mod_h
    # crown: two small stepped tiers
    wc = mod_w_top * 0.82
    p += [(z, wc), (z + 3.0, wc + 0.3), (z + 3.0, wc * 0.8), (z + 5.5, wc * 0.8 + 0.2)]
    z += 5.5
    if spire:  # short, chunky spire with a ball tip (≈5 mm thick, won't snap)
        p += [(z, 2.8), (z + 12, 2.2)]
        z += 12
        for a in np.linspace(-0.3, np.pi / 2, 10):
            p.append((z + 2.4 * np.sin(a) + 0.7, 2.6 * np.cos(a) + 1e-3))
    else:      # rounded dome cap instead of a spire
        r = wc * 0.8 + 0.2
        for a in np.linspace(0, np.pi / 2, 12):
            p.append((z + 0.8 * r * np.sin(a), r * np.cos(a) + 1e-3))
    return p


# ---------- mesh helpers ----------
def loft(profile, shape):
    quads = []
    for (z1, w1), (z2, w2) in zip(profile[:-1], profile[1:]):
        a, b = shape * w1, shape * w2
        for i in range(len(shape)):
            j = (i + 1) % len(shape)
            quads.append([(*a[i], z1), (*a[j], z1), (*b[j], z2), (*b[i], z2)])
    z0, w0 = profile[0]
    quads.append([(*pt, z0) for pt in shape * w0])
    return quads


def medallions(z, face_w, r, depth=0.9, sides=4):
    """Ruyi-style round medallion: disc with a raised ring, one per face."""
    polys, tones = [], []
    t = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    for k in range(sides):
        ang = k * 2 * np.pi / sides
        nrm = np.array([np.cos(ang), np.sin(ang), 0.0])
        tan = np.array([-np.sin(ang), np.cos(ang), 0.0])
        up = np.array([0, 0, 1.0])
        c0 = nrm * face_w + up * z
        for rr, d, tone in [(r, depth, 1.0), (r * 0.62, depth + 0.5, 0.85), (r * 0.3, depth + 0.8, 1.0)]:
            c = c0 + nrm * d
            ring = [c + rr * (np.cos(a) * tan + np.sin(a) * up) for a in t]
            polys.append(ring); tones.append(tone)
            for i in range(len(t)):  # rim
                j = (i + 1) % len(t)
                p1, p2 = ring[i], ring[j]
                polys.append([p1, p2, p2 - nrm * (d - 0.2), p1 - nrm * (d - 0.2)]); tones.append(0.8)
    return polys, tones


def tray(w_out, h, wall):
    prof = [(0, w_out), (h, w_out), (h, w_out - wall), (2.5, w_out - wall), (2.5, 0.01)]
    return loft(prof, chamfered_square(0.12))


def render(ax, quads, color, extra=(), extra_color=None, view=(9, -52), lim=26, H=138):
    polys = [np.array(q, float) for q in quads]
    cols = []
    base = np.array(matplotlib.colors.to_rgb(color))
    def shade(p, b, tone=1.0):
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9 and len(p) > 3:
            n = np.cross(p[2] - p[0], p[3] - p[0])
        n = n / (np.linalg.norm(n) + 1e-12)
        lam = abs(n @ LIGHT)
        spec = lam ** 18 * 0.35
        return np.clip(b * tone * (0.28 + 0.72 * lam) + spec, 0, 1)
    for p in polys:
        cols.append(shade(p, base))
    if extra:
        eb = np.array(matplotlib.colors.to_rgb(extra_color or color))
        for p, tone in zip(*extra):
            p = np.array(p, float); polys.append(p); cols.append(shade(p, eb, tone))
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.15)
    ax.add_collection3d(pc)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, H)
    ax.set_box_aspect((1, 1, H / (2 * lim)), zoom=1.0); ax.view_init(*view); ax.set_axis_off()


def real_101(ax, color):
    """Schematic elevation of the real Taipei 101 for side-by-side comparison."""
    s = 1 / 4.6  # metres -> plot units
    prof = [(0, 40), (100, 25)]
    z = 100; prof.append((z, 23))
    for i in range(8):
        prof += [(z, 24), (z + 33, 27.5)]; z += 33
    prof += [(z, 19), (z + 20, 20), (z + 20, 14), (z + 34, 15), (z + 34, 5), (z + 38, 5), (z + 38, 2.5), (508, 0.6)]
    zz = np.array([p[0] for p in prof]) * s; ww = np.array([p[1] for p in prof]) * s
    ax.fill_betweenx(zz, -ww, ww, color=color, lw=0)
    for i in range(8):  # module lines
        zl = (100 + 33 * i) * s
        ax.plot([-6, 6], [zl, zl], color="w", lw=0.4)
    for x in (-3.2, 3.2):
        ax.add_patch(plt.Circle((x, 92 * s), 1.3, color="w", fill=False, lw=0.8))
    ax.set_aspect("equal"); ax.set_xlim(-30, 30); ax.set_ylim(-3, 125); ax.axis("off")
    ax.set_title("real Taipei 101\n(schematic)", fontsize=10, color="#555")


GLASS, JADE, STONE, GOLD, BRONZE = "#4f9c94", "#3f8a6e", "#9aa3a8", "#c8a24a", "#9a6b4b"

# Ring cone math: a round ring rests where it spans the section's outer corners.
CONCEPTS = []

def podium_face(p, z):
    """Half-width of the sloped podium at height z (for placing medallions)."""
    (z0, w0), (z1, w1) = p[2], p[3]
    return w0 + (w1 - w0) * (z - z0) / (z1 - z0)


PH = 28  # podium height
# 1. Classic 101 – faithful square plan, notched corners, podium, medallions, spire stub
p1 = tower_profile(0, 17, PH, 9.0, 5.8, 1.0)
CONCEPTS.append(("1_classic_101", "1. Classic 101", loft(p1, notched_square()),
                 medallions(PH - 6, podium_face(p1, PH - 6), 3.4, 0.6), GLASS, GOLD))

# 2. Round 101 – revolved (round) so every ring sits evenly; same tiers/podium/medallions
p2 = tower_profile(0, 17, PH, 8.4, 5.2, 1.1)
CONCEPTS.append(("2_round_101", "2. Round 101", loft(p2, circle()),
                 medallions(PH - 6, podium_face(p2, PH - 6) - 0.4, 3.4, 0.6), JADE, GOLD))

# 3. Soft 101 – chamfered corners, dome cap instead of spire (no pokey bits at all)
p3 = tower_profile(0, 17, PH, 8.8, 5.6, 1.0, spire=False)
CONCEPTS.append(("3_soft_101", "3. Soft 101", loft(p3, chamfered_square()),
                 medallions(PH - 6, podium_face(p3, PH - 6), 3.4, 0.6), STONE, BRONZE))

# 4. 101 on a tray – tower rises from a square catch-all tray (earrings, cufflinks)
p4 = [(z + 2.5, w) for z, w in tower_profile(0, 15, PH - 4, 9.0, 5.8, 1.0)]
q4 = tray(36, 9, 2.4) + loft(p4, notched_square())
CONCEPTS.append(("4_tray_101", "4. 101 on a Tray", q4,
                 medallions(PH - 8.5, podium_face(p4, PH - 8.5), 3.2, 0.6), GLASS, GOLD))

# 5. Bold 101 – exaggerated flare so the stacked "boxes" pop; heavier podium
p5 = tower_profile(0, 19, PH, 8.4, 5.0, 1.8, mod_h=10)
CONCEPTS.append(("5_bold_101", "5. Bold 101", loft(p5, notched_square(0.12)),
                 medallions(PH - 6, podium_face(p5, PH - 6), 3.8, 0.6), GOLD, "#7a5b1e"))


def ring_marks(ax, profile, shape, color, zmin=30):
    """Side profile + where ring sizes 4 / 8 / 13 come to rest (corner span)."""
    corner = np.max(np.linalg.norm(shape, axis=1))
    zs = np.array([p[0] for p in profile]); ws = np.array([p[1] for p in profile])
    ax.fill_betweenx(zs, -ws, ws, color=color, lw=0, alpha=0.9)
    for d, lab in [(14.9, "size 4"), (18.2, "size 8"), (22.2, "size 13")]:
        # highest point whose corner diameter >= ring diameter (above podium)
        cand = [(z, w) for z, w in profile if z > zmin and 2 * w * corner >= d]
        if cand:
            zr = max(c[0] for c in cand)
            ax.plot([-d / 2, d / 2], [zr, zr], color="#c0392b", lw=1.6)
            ax.text(d / 2 + 2, zr, lab, fontsize=7, va="center", color="#c0392b")
    ax.set_aspect("equal"); ax.set_xlim(-42, 48); ax.set_ylim(-2, 140)
    ax.tick_params(labelsize=6); ax.set_xlabel("mm (front elevation)", fontsize=7)


if __name__ == "__main__":
    profiles = [(p1, notched_square()), (p2, circle()), (p3, chamfered_square()),
                (p4, notched_square()), (p5, notched_square(0.12))]
    fig_all = plt.figure(figsize=(22, 7.2), dpi=110)
    a = fig_all.add_subplot(1, 6, 1); real_101(a, "#5c8f8a")
    for i, ((slug, title, quads, med, color, mcol), (prof, shp)) in enumerate(zip(CONCEPTS, profiles)):
        fig = plt.figure(figsize=(11, 6.5), dpi=120)
        r0 = fig.add_subplot(1, 3, 1); real_101(r0, "#5c8f8a")
        a1 = fig.add_subplot(1, 3, 2, projection="3d"); render(a1, quads, color, med, mcol)
        a2 = fig.add_subplot(1, 3, 3); ring_marks(a2, prof, shp, color)
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(); fig.savefig(f"{OUT}/{slug}.png", bbox_inches="tight"); plt.close(fig)
        ax = fig_all.add_subplot(1, 6, i + 2, projection="3d"); render(ax, quads, color, med, mcol)
        ax.set_title(title, fontsize=13)
    fig_all.tight_layout(); fig_all.savefig(f"{OUT}/all_concepts_v2.png", bbox_inches="tight")
