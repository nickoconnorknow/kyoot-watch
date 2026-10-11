"""Printable "101 on a Tray" ring stand (no medallions), <= 3 in tall. Units: mm.

Writes taipei101_tray_ring_stand.3mf (+ .stl) and a preview PNG.
"""
import os, sys
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
from render import notched_square, chamfered_square  # noqa: E402

# ---- dimensions ----
TRAY_W, TRAY_H, TRAY_WALL, TRAY_FLOOR = 34.0, 8.0, 2.4, 2.5   # 68 mm square tray
PODIUM_W, PODIUM_H = 15.0, 12.0
MOD_H, N_MOD, FLARE = 6.0, 8, 1.0
MOD_W_BOT, MOD_W_TOP = 8.8, 5.6


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
    tray = solid_loft([(0, TRAY_W), (TRAY_H, TRAY_W), (TRAY_H, TRAY_W - TRAY_WALL),
                       (TRAY_FLOOR, TRAY_W - TRAY_WALL)], chamfered_square(0.12))
    tower = solid_loft(tower_profile(TRAY_FLOOR), notched_square())
    for name, m in (("tray", tray), ("tower", tower)):
        assert m.is_watertight and m.is_volume, f"{name} not a closed volume"
    return trimesh.boolean.union([tray, tower], engine="manifold")


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

    # preview
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    L = np.array([-0.55, -0.75, 0.6]); L /= np.linalg.norm(L)
    tri = mesh.vertices[mesh.faces]
    shade = 0.3 + 0.7 * np.clip(mesh.face_normals @ L, 0, 1)
    cols = np.clip(shade[:, None] * np.array(matplotlib.colors.to_rgb("#4f9c94")), 0, 1)
    fig = plt.figure(figsize=(9, 6), dpi=130)
    for k, view in enumerate([(14, -52), (40, -30)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors=cols, linewidths=0.1))
        ax.set_xlim(-36, 36); ax.set_ylim(-36, 36); ax.set_zlim(0, 76)
        ax.set_box_aspect((1, 1, 76 / 72)); ax.view_init(*view); ax.set_axis_off()
    fig.suptitle(f'Taipei 101 tray ring stand — {size[2]:.0f} mm ({size[2]/25.4:.2f}") tall, '
                 f'{size[0]:.0f} mm ({size[0]/25.4:.2f}") square', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "preview.png"), bbox_inches="tight")
