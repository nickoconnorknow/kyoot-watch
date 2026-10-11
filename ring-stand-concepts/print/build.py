"""Printable "101 on a Tray" ring stand (no medallions), <= 3 in tall. Units: mm.

Writes taipei101_tray_ring_stand.3mf (+ .stl) and a preview PNG.
"""
import os, sys
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
from render import notched_square  # noqa: E402

# ---- dimensions ----
TRAY_W, TRAY_H, TRAY_WALL, TRAY_FLOOR = 32.0, 7.0, 2.4, 2.5   # 64 mm square tray
TRAY_R = 9.0                                                  # smooth rounded corners
PODIUM_W, PODIUM_H = 12.0, 12.0
MOD_H, N_MOD, FLARE = 6.0, 8, 0.9
MOD_W_BOT, MOD_W_TOP = 7.7, 4.7                               # slimmer tower


def solid_loft(profile, shape):
    """Watertight mesh: stack of rings (z, half_width) of `shape`, fan-capped at both ends."""
    keep = np.linalg.norm(shape - np.roll(shape, 1, axis=0), axis=1) > 1e-9
    shape = shape[keep]                      # drop duplicated corner points
    m = len(shape)
    verts, faces = [], []
    for z, w in profile:
        verts += [(x * w, y * w, z) for x, y in shape]
    for k in range(len(profile) - 1):
        a, b = k * m, (k + 1) * m
        for i in range(m):
            j = (i + 1) % m
            faces += [(a + i, a + j, b + j), (a + i, b + j, b + i)]
    bot = len(verts); verts.append((0, 0, profile[0][0]))
    top = len(verts); verts.append((0, 0, profile[-1][0]))
    last = (len(profile) - 1) * m
    for i in range(m):
        j = (i + 1) % m
        faces.append((bot, j, i))
        faces.append((top, last + i, last + j))
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def rounded_square(half, r, seg=16):
    pts = []
    for cx, cy, a0 in [(1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)]:
        for a in np.radians(np.linspace(a0, a0 + 90, seg)):
            pts.append((cx * (half - r) + r * np.cos(a), cy * (half - r) + r * np.sin(a)))
    return pts


def to_manifold(mesh):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                               tri_verts=np.asarray(mesh.faces, np.uint32)))


def tray_manifold():
    import manifold3d as mf
    outer = mf.CrossSection([rounded_square(TRAY_W, TRAY_R)]).extrude(TRAY_H)
    inner = (mf.CrossSection([rounded_square(TRAY_W - TRAY_WALL, TRAY_R - TRAY_WALL)])
             .extrude(TRAY_H).translate((0, 0, TRAY_FLOOR)))
    return outer - inner


def tower_profile(z0):
    z = z0 + PODIUM_H
    p = [(z0 - 0.6, PODIUM_W), (z, MOD_W_BOT + FLARE + 1.4)]     # sloped podium (sunk 0.6 into floor)
    for i in range(N_MOD):
        w = MOD_W_BOT + (MOD_W_TOP - MOD_W_BOT) * i / (N_MOD - 1)
        p += [(z, w), (z + MOD_H, w + FLARE)]                    # flared module = ring ledge
        z += MOD_H
    wc = MOD_W_TOP * 0.82
    p += [(z, wc), (z + 2.0, wc + 0.25), (z + 2.0, wc * 0.8), (z + 3.5, wc * 0.8 + 0.15)]
    z += 3.5
    p += [(z, 2.7), (z + 3.5, 2.3)]                              # short thick spire
    z += 3.5
    for a in np.linspace(-0.3, np.pi / 2 - 0.25, 9):             # ball tip
        p.append((z + 2.3 * np.sin(a) + 0.6, 2.5 * np.cos(a)))
    return p


def build():
    tower = solid_loft(tower_profile(TRAY_FLOOR), notched_square())
    assert tower.is_watertight and tower.is_volume, "tower not a closed volume"
    out = (tray_manifold() + to_manifold(tower)).to_mesh()
    return trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts)


def zbuffer(mesh, elev, azim, px=700):
    e, a = np.radians(elev), np.radians(azim)
    R1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    v = mesh.vertices @ R1.T @ R2.T      # x right, y depth (toward viewer = -), z up
    n = mesh.face_normals @ R1.T @ R2.T
    L = np.array([-0.45, -0.7, 0.55]); L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0, 1); spec = np.clip(n @ np.array([0, -1, 0]), 0, 1) ** 30 * 0.25
    base = np.array([0.31, 0.61, 0.58])
    col = np.clip((0.25 + 0.75 * lam)[:, None] * base + spec[:, None], 0, 1)
    s = (px * 0.78) / max(np.ptp(v[:, 0]), np.ptp(v[:, 2]))
    xy = np.c_[(v[:, 0] - v[:, 0].mean()) * s + px / 2, px / 2 - (v[:, 2] - v[:, 2].mean()) * s]
    depth = v[:, 1]
    img = np.ones((px, px, 3)); zb = np.full((px, px), np.inf)
    for f, c in zip(mesh.faces, col):
        p = xy[f]; d = depth[f]
        x0, y0 = np.floor(p.min(0)).astype(int); x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, px - 1), min(y1, px - 1)
        if x1 < x0 or y1 < y0: continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12: continue
        w0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        w1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        w2 = 1 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not m.any(): continue
        z = w0 * d[0] + w1 * d[1] + w2 * d[2]
        sub = zb[y0:y1 + 1, x0:x1 + 1]
        upd = m & (z < sub)
        sub[upd] = z[upd]
        img[y0:y1 + 1, x0:x1 + 1][upd] = c
    return img


def preview(mesh, size):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.4), dpi=130)
    for ax, (el, az) in zip(axs, [(12, 35), (38, 20)]):
        ax.imshow(zbuffer(mesh, el, az)); ax.axis("off")
    fig.suptitle(f'Taipei 101 tray ring stand: {size[2]:.0f} mm ({size[2]/25.4:.2f}") tall, '
                 f'{size[0]:.0f} mm ({size[0]/25.4:.2f}") square', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "preview.png"), bbox_inches="tight")


if __name__ == "__main__":
    mesh = build()
    assert mesh.is_watertight and mesh.is_volume
    lo, hi = mesh.bounds
    size = hi - lo
    print(f"size: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm  "
          f"({size[0]/25.4:.2f} x {size[1]/25.4:.2f} x {size[2]/25.4:.2f} in)")
    print(f"volume {mesh.volume/1000:.1f} cm^3, faces {len(mesh.faces)}")
    mesh.export(os.path.join(HERE, "taipei101_tray_ring_stand.3mf"))
    mesh.export(os.path.join(HERE, "taipei101_tray_ring_stand.stl"))

    preview(mesh, size)
