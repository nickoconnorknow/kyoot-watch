"""Quick concept renders for Taipei 101-inspired ring stands (units: mm)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

N_T, N_Z = 160, 700
ls = LightSource(azdeg=315, altdeg=35)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def tier_saw(u, soft=0.15):
    """0..1 ramp within each tier, with a rounded (not knife-edge) drop."""
    f = u % 1.0
    rise = f / (1 - soft)
    fall = 1 - smoothstep((f - (1 - soft)) / soft)
    return np.where(f < 1 - soft, rise, fall)


def superellipse(theta, n):
    """Radius multiplier for a rounded square (n=2 circle, n->inf square)."""
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    return (c ** n + s ** n) ** (-1.0 / n)


# Each concept: profile(z, theta) -> radius. z in mm.
def pagoda_cone(z, t):
    H = 105
    env = 23 - 16.5 * (z / H)                      # straight cone envelope
    base = np.where(z < 14, 27 - 1.5 * smoothstep(z / 14), 0)
    tiers = (z >= 14) & (z < 92)
    u = (z - 14) / (78 / 8)
    r = np.where(tiers, env + 2.2 * tier_saw(u, 0.25), env)
    top = z >= 92                                   # rounded bulb tip, no spire
    r = np.where(top, np.sqrt(np.clip(13 ** 2 - (z - 92) ** 2, 0, None)) * 0.55, r)
    r = np.where(z < 14, base, r)
    return r


def soft_sweep(z, t):
    H = 110
    env = 22 * (1 - z / H) ** 0.85 + 5
    ripple = 1.2 * np.sin(np.pi * ((z - 16) / 9.5) % np.pi) ** 2
    r = env + np.where((z > 16) & (z < 92), ripple, 0)
    r = np.where(z < 16, 30 - 3 * smoothstep(z / 16), r)
    cap = z > 98
    r = np.where(cap, 7.2 * np.sqrt(np.clip(1 - ((z - 98) / 12) ** 2, 0, None)), r)
    return r


def square_tiers(z, t):
    H = 108
    sq = superellipse(t, 4.5)                       # rounded-square section
    env = 20 - 13.5 * (z / H)
    u = (z - 18) / (76 / 8)
    tiers = (z >= 18) & (z < 94)
    r = np.where(tiers, env + 2.6 * tier_saw(u, 0.3), env) * sq
    r = np.where(z < 18, (30 - 4 * smoothstep(z / 18)) * superellipse(t, 3), r)
    cap = z >= 94
    r = np.where(cap, 8.5 * np.sqrt(np.clip(1 - ((z - 94) / 13) ** 2, 0, None)) * sq, r)
    return r


def pearl_dish(z, t):
    # Wide shallow dish base (holds earrings), slim tiered cone, pearl on top.
    dish = np.where(z < 4, 40, 40 - 0 * z)
    dish = np.where(z < 10, 40 - 12 * smoothstep((z - 4) / 6) * (z > 4), 0)
    dish = np.where(z < 10, np.maximum(dish, 28), 0)
    env = 15.5 - 9 * ((z - 10) / 85)
    u = (z - 10) / (80 / 8)
    stem = env + 1.6 * tier_saw(u, 0.35)
    r = np.where(z < 10, dish, stem)
    r = np.where((z >= 10) & (z < 13), 17, r)       # little podium collar
    pearl_c = 99
    neck = (z >= 90) & (z < pearl_c - 6)
    r = np.where(neck, 6.3, r)
    pearl = z >= pearl_c - 6
    r = np.where(pearl, np.sqrt(np.clip(9.5 ** 2 - (z - pearl_c) ** 2, 0, None)), r)
    return r


def octagon_minimal(z, t):
    H = 110
    k = 8
    oct_ = np.cos(np.pi / k) / np.cos(((t + np.pi / k) % (2 * np.pi / k)) - np.pi / k)
    oct_ = oct_ ** 0.55                              # soften facets a bit
    env = 24 - 17 * (z / H)
    groove = np.zeros_like(z)
    for i in range(1, 8):
        zc = 14 + i * 10
        groove -= 0.9 * np.exp(-((z - zc) / 0.9) ** 2)
    r = (env + groove) * oct_
    r = np.where(z < 8, 28 * oct_, r)
    cap = z > 94
    r = np.where(cap, 6.8 * np.sqrt(np.clip(1 - ((z - 94) / 12) ** 2, 0, None)) * oct_, r)
    return r


CONCEPTS = [
    ("1_pagoda_cone", "1. Pagoda Cone", pagoda_cone, 106, "#c9a227"),
    ("2_soft_sweep", "2. Soft Sweep", soft_sweep, 110, "#7fa7a0"),
    ("3_square_tiers", "3. Square Tiers", square_tiers, 107, "#6f8fb5"),
    ("4_pearl_dish", "4. Pearl & Dish", pearl_dish, 108.5, "#b07a8c"),
    ("5_octagon_minimal", "5. Octagon Minimal", octagon_minimal, 106, "#8c8c8c"),
]


def draw(ax, fn, h, color):
    z = np.linspace(0, h, N_Z)
    t = np.linspace(0, 2 * np.pi, N_T)
    Z, T = np.meshgrid(z, t)
    R = fn(Z, T)
    X, Y = R * np.cos(T), R * np.sin(T)
    rgb = ls.shade(Z * 0 + 1, plt.cm.colors.ListedColormap([color]), vert_exag=0, blend_mode="soft")
    # simple lambert-ish shading from surface normals
    dx = np.gradient(X, axis=1), np.gradient(X, axis=0)
    dy = np.gradient(Y, axis=1), np.gradient(Y, axis=0)
    dz = np.gradient(Z, axis=1), np.gradient(Z, axis=0)
    n = np.cross(np.stack([dx[0], dy[0], dz[0]], -1), np.stack([dx[1], dy[1], dz[1]], -1))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    light = np.array([-0.5, -0.7, 0.5]); light /= np.linalg.norm(light)
    shade = np.clip(np.abs(n @ light), 0, 1) * 0.75 + 0.25
    base = np.array(matplotlib.colors.to_rgb(color))
    fc = np.clip(shade[..., None] * base, 0, 1)
    ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=2, linewidth=0, antialiased=False, shade=False)
    m = 42
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(0, 2 * m * 1.3)
    ax.set_box_aspect((1, 1, 1.3))
    ax.view_init(elev=14, azim=-60)
    ax.set_axis_off()


def profile(ax, fn, h, color):
    z = np.linspace(0, h, N_Z)
    r = fn(z, np.zeros_like(z))
    ax.fill_betweenx(z, -r, r, color=color, alpha=0.85, lw=0)
    for d, lab in [(14.9, "size 4"), (18.2, "size 8"), (22.2, "size 13")]:
        zz = z[np.argmin(np.abs(np.where((z > 15) & (z < 88), 2 * r, 1e9) - d))]
        ax.plot([-d / 2 - 4, d / 2 + 4], [zz, zz], color="k", lw=0.6)
        ax.text(d / 2 + 5, zz, lab, fontsize=6, va="center")
    ax.set_aspect("equal"); ax.set_xlim(-42, 50); ax.set_ylim(-2, 115)
    ax.tick_params(labelsize=6); ax.set_xlabel("mm", fontsize=7)


if __name__ == "__main__":
    import os
    out = os.path.dirname(os.path.abspath(__file__))
    fig_all = plt.figure(figsize=(20, 6.5), dpi=110)
    for i, (slug, title, fn, h, color) in enumerate(CONCEPTS):
        fig = plt.figure(figsize=(8, 6), dpi=120)
        a1 = fig.add_subplot(1, 2, 1, projection="3d"); draw(a1, fn, h, color)
        a2 = fig.add_subplot(1, 2, 2); profile(a2, fn, h, color)
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(); fig.savefig(f"{out}/{slug}.png"); plt.close(fig)
        a = fig_all.add_subplot(1, 5, i + 1, projection="3d"); draw(a, fn, h, color)
        a.set_title(title, fontsize=13)
    fig_all.tight_layout(); fig_all.savefig(f"{out}/all_concepts.png", bbox_inches="tight")
