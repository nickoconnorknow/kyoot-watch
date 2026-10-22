"""Printable "101 on a Tray" ring stand (no medallions), <= 3 in tall. Units: mm.

Writes taipei101_tray_ring_stand.3mf (+ .stl) and a preview PNG.

The tower is nearly straight like the real building. Each of the 8 flared
modules ends in a small ledge; ledges step down in size by LEDGE_STEP mm going
up, and the ring is aimed to come to rest on ledge CATCH_MODULE. Set RING_SIZE
to the wearer's US ring size and rebuild.
"""
import os, sys
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v2"))
from render import notched_square  # noqa: E402

# ---- ring target ----
RING_SIZE = 9.0                      # US size; inner diameter = 11.63 + 0.8128 * size
CATCH_MODULE = 5                     # 0 = bottom module ... 7 = top module
LEDGE_STEP = 0.6                     # ledge corner-diameter change per module (mm) -> gentle taper

# ---- dimensions ----
TRAY_W, TRAY_H, TRAY_WALL, TRAY_FLOOR = 28.0, 6.5, 2.4, 2.5   # 56 mm square tray
TRAY_R = 8.0                                                  # rounded corners
PODIUM_H = 10.0
MOD_H, N_MOD, FLARE = 6.6, 8, 0.8
NOTCH = 0.12                 # corner notch step (fraction of half-width): wide enough to wipe out
FILLET_IN, FILLET_OUT = 0.05, 0.035   # inside / outside corner rounding (fraction, ~0.35/0.25 mm)
LIP = 0.25                   # chamfer on each ledge lip (mm)
STEP_DZ = 0.5                # ledges slope down-inward over this height instead of a sharp shelf
RING_BAND, RING_THICK = 6.0, 2.0      # mockup ring: band width, wall thickness (mm)


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


def ring_loft(rings):
    """Watertight mesh from a stack of convex rings [(pts(m,2), z)], fan-capped at both ends."""
    m = len(rings[0][0])
    verts, faces = [], []
    for pts, z in rings:
        verts += [(x, y, z) for x, y in pts]
    for k in range(len(rings) - 1):
        a, b = k * m, (k + 1) * m
        for i in range(m):
            j = (i + 1) % m
            faces += [(a + i, a + j, b + j), (a + i, b + j, b + i)]
    bot = len(verts); verts.append((0, 0, rings[0][1]))
    top = len(verts); verts.append((0, 0, rings[-1][1]))
    last = (len(rings) - 1) * m
    for i in range(m):
        j = (i + 1) % m
        faces += [(bot, j, i), (top, last + i, last + j)]
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    mesh.fix_normals()
    return mesh


def tray_mesh():
    """Rounded-square tray; the wall cross-section (inset d, height z) is swept around the
    outline, giving a fully rounded top lip, a soft inside floor fillet and a small
    bottom-edge chamfer (helps against elephant's foot)."""
    W, H, t, fl = TRAY_W, TRAY_H, TRAY_WALL, TRAY_FLOOR
    rl, fi = t / 2, 1.5                                   # lip radius, floor fillet radius
    sec = [(0.6, 0.0), (0.0, 0.6), (0.0, H - rl)]
    sec += [(rl - rl * np.cos(a), H - rl + rl * np.sin(a)) for a in np.linspace(0, np.pi, 13)[1:]]
    sec += [(t + fi - fi * np.cos(a), fl + fi - fi * np.sin(a)) for a in np.linspace(0, np.pi / 2, 7)]
    rings = [(np.array(rounded_square(W - d, TRAY_R - d)), z) for d, z in sec]
    return ring_loft(rings)


def smooth_section():
    """Notched-square plan with rounded inside and outside corners (no sharp crevices)."""
    import manifold3d as mf
    k = 10.0  # work at 10x scale for offset precision
    pts = notched_square(NOTCH)
    keep = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1) > 1e-9
    cs = mf.CrossSection([(pts[keep] * k).tolist()])
    cs = cs.offset(FILLET_IN * k, mf.JoinType.Round, 2.0, 96).offset(-FILLET_IN * k, mf.JoinType.Round, 2.0, 96)
    cs = cs.offset(-FILLET_OUT * k, mf.JoinType.Round, 2.0, 96).offset(FILLET_OUT * k, mf.JoinType.Round, 2.0, 96)
    poly = max(cs.to_polygons(), key=len)
    return np.asarray(poly, float) / k


def ring_diameter(size):
    return 11.63 + 0.8128 * size


def tower_profile(z0, shape):
    """Returns (profile, ledge half-widths, ledge heights, corner factor)."""
    corner = np.max(np.linalg.norm(shape, axis=1))       # ring touches the (rounded) corners
    d = ring_diameter(RING_SIZE)
    ledge = [(d + LEDGE_STEP * (CATCH_MODULE - i + 0.5)) / (2 * corner) for i in range(N_MOD)]
    pod_top = ledge[0] + 1.0
    pod_bot, r = pod_top + 2.6, 1.5
    p = [(z0 - 0.6, pod_bot + r)]
    # concave cove where the podium meets the tray floor
    p += [(z0 + r - r * np.cos(a), pod_bot + r - r * np.sin(a)) for a in np.linspace(0, np.pi / 2, 7)]

    def step(zt, w_lip, w_next, dz=STEP_DZ, lip=LIP):
        # softened lip, then a sloped (not flat) ledge down to the next wall
        return [(zt - lip, w_lip), (zt, w_lip - lip), (zt + dz, w_next)]

    zt = z0 + PODIUM_H
    p += step(zt, pod_top, ledge[0] - FLARE)
    zl = []
    for i in range(N_MOD):
        zt += MOD_H
        nxt = ledge[i + 1] - FLARE if i + 1 < N_MOD else ledge[-1] * 0.8
        p += step(zt, ledge[i], nxt)
        zl.append(zt - LIP)
    wc = ledge[-1] * 0.8
    zt += 2.4; p += step(zt, wc + 0.2, wc * 0.8, 0.4, 0.2)
    zt += 1.8; p += step(zt, wc * 0.8 + 0.15, 2.5, 0.5, 0.2)
    z = zt + 0.5
    p += [(z + 2.6, 2.2)]                                       # short thick spire
    z += 2.6
    for a in np.linspace(-0.3, np.pi / 2 - 0.25, 9):            # ball tip
        p.append((z + 2.2 * np.sin(a) + 0.6, 2.4 * np.cos(a)))
    return p, ledge, zl, corner


def ring_mesh(z_seat):
    """Mockup ring: plain band with rounded outer edges, bottom inner edge on the ledge lip."""
    d = ring_diameter(RING_SIZE)
    ri, ro, h, rr = d / 2, d / 2 + RING_THICK, RING_BAND, 0.8
    sec = [(ri, 0.0), (ri, h)]
    sec += [(ro - rr + rr * np.sin(a), h - rr + rr * np.cos(a)) for a in np.linspace(0, np.pi / 2, 5)]
    sec += [(ro - rr + rr * np.sin(a), rr - rr * np.cos(a) * -1 + 0) for a in []]
    sec += [(ro - rr + rr * np.cos(a), rr - rr * np.sin(a)) for a in np.linspace(0, np.pi / 2, 5)]
    t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    m, n = len(sec), len(t)
    V = np.array([(R * np.cos(a), R * np.sin(a), z + z_seat) for a in t for R, z in sec])
    F = []
    for i in range(n):
        for j in range(m):
            a, b = i * m + j, ((i + 1) % n) * m + j
            a2, b2 = i * m + (j + 1) % m, ((i + 1) % n) * m + (j + 1) % m
            F += [(a, b, b2), (a, b2, a2)]
    ring = trimesh.Trimesh(V, np.array(F), process=True); ring.fix_normals()
    return ring


def build():
    shape = smooth_section()
    prof, ledge, zl, corner = tower_profile(TRAY_FLOOR, shape)
    tower = solid_loft(prof, shape)
    tray = tray_mesh()
    for name, m in (("tray", tray), ("tower", tower)):
        assert m.is_watertight and m.is_volume, f"{name} not a closed volume"
    out = (to_manifold(tray) + to_manifold(tower)).to_mesh()
    d = ring_diameter(RING_SIZE)
    print(f"ring size {RING_SIZE:g} = {d:.2f} mm inner diameter")
    for i, w in enumerate(ledge):
        D = 2 * w * corner
        sz = (D - 11.63) / 0.8128
        tag = "  <- ring rests here" if i == CATCH_MODULE else ""
        print(f"  ledge {i + 1}: {D:5.2f} mm corner span (~size {sz:4.1f}), "
              f"{zl[i]:5.1f} mm up, flats {2 * w:5.2f} mm{tag}")
    mesh = trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts)
    return mesh, prof, zl[CATCH_MODULE]


def zbuffer(parts, elev, azim, px=700, frame=None):
    """parts: [(mesh, rgb)]. frame: mesh used for framing (default: all)."""
    mesh = trimesh.util.concatenate([m for m, _ in parts])
    fbase = np.concatenate([np.tile(c, (len(m.faces), 1)) for m, c in parts])
    e, a = np.radians(elev), np.radians(azim)
    R1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    R2 = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    v = mesh.vertices @ R1.T @ R2.T      # x right, y depth (toward viewer = -), z up
    n = mesh.face_normals @ R1.T @ R2.T
    L = np.array([-0.45, -0.7, 0.55]); L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0, 1); spec = np.clip(n @ np.array([0, -1, 0]), 0, 1) ** 30 * 0.25
    col = np.clip((0.25 + 0.75 * lam)[:, None] * fbase + spec[:, None], 0, 1)
    fv = (frame.vertices if frame is not None else mesh.vertices) @ R1.T @ R2.T
    s = (px * 0.82) / max(np.ptp(fv[:, 0]), np.ptp(fv[:, 2]))
    cx0, cz0 = (fv[:, 0].max() + fv[:, 0].min()) / 2, (fv[:, 2].max() + fv[:, 2].min()) / 2
    xy = np.c_[(v[:, 0] - cx0) * s + px / 2, px / 2 - (v[:, 2] - cz0) * s]
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


TEAL, GOLD = np.array([0.31, 0.61, 0.58]), np.array([0.85, 0.66, 0.25])


def preview(mesh, size, prof, z_seat):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ring = ring_mesh(z_seat)
    d = ring_diameter(RING_SIZE)
    fig = plt.figure(figsize=(15, 6.2), dpi=130)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    ax = fig.add_subplot(gs[0]); ax.imshow(zbuffer([(mesh, TEAL), (ring, GOLD)], 12, 35, px=800)); ax.axis("off")
    ax.set_title("with a size %g ring" % RING_SIZE, fontsize=11)
    ax = fig.add_subplot(gs[1])
    box = trimesh.creation.box(extents=(26, 26, 26)); box.apply_translation((0, 0, z_seat + 2))
    ax.imshow(zbuffer([(mesh, TEAL), (ring, GOLD)], 22, 35, px=800, frame=box)); ax.axis("off")
    ax.set_title("close-up: ring resting on ledge %d" % (CATCH_MODULE + 1), fontsize=11)
    ax = fig.add_subplot(gs[2])
    zz = np.array([q[0] for q in prof]); ww = np.array([q[1] for q in prof])
    corner = np.max(np.linalg.norm(smooth_section(), axis=1))
    lt = TEAL * 0.55 + 0.45
    ax.fill_between([-TRAY_W, TRAY_W], 0, TRAY_FLOOR, color=lt, lw=0)
    for x in (-TRAY_W, TRAY_W - TRAY_WALL):
        ax.add_patch(plt.Rectangle((x, 0), TRAY_WALL, TRAY_H, color=lt, lw=0))
    ax.fill_betweenx(zz, -ww * corner, ww * corner, color=lt, lw=0, label="corners")
    ax.fill_betweenx(zz, -ww, ww, color=TEAL, lw=0, label="flat faces")
    ax.add_patch(plt.Rectangle((d / 2, z_seat), RING_THICK, RING_BAND, color=GOLD, label="size %g ring" % RING_SIZE))
    ax.add_patch(plt.Rectangle((-d / 2 - RING_THICK, z_seat), RING_THICK, RING_BAND, color=GOLD))
    ax.axhline(z_seat, color="#b5462e", lw=0.8, ls="--")
    ax.text(-TRAY_W + 1, z_seat - 4, 'ring sits %.0f mm (%.2f") up' % (z_seat, z_seat / 25.4), fontsize=8, color="#b5462e")
    ax.set_aspect("equal"); ax.set_xlim(-TRAY_W - 1, TRAY_W + 1); ax.set_ylim(-1, size[2] + 3)
    ax.set_xlabel("mm", fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc="upper right")
    ax.set_title("side view", fontsize=11)
    fig.suptitle('Taipei 101 ring stand, size %g ring: %.0f mm (%.2f") tall, %.0f mm (%.2f") square tray'
                 % (RING_SIZE, size[2], size[2] / 25.4, size[0], size[0] / 25.4), fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "preview.png"), bbox_inches="tight")


if __name__ == "__main__":
    mesh, prof, z_seat = build()
    assert mesh.is_watertight and mesh.is_volume
    lo, hi = mesh.bounds
    size = hi - lo
    print(f"size: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm  "
          f"({size[0]/25.4:.2f} x {size[1]/25.4:.2f} x {size[2]/25.4:.2f} in)")
    print(f"volume {mesh.volume/1000:.1f} cm^3, faces {len(mesh.faces)}")
    mesh.export(os.path.join(HERE, "taipei101_tray_ring_stand.3mf"))
    mesh.export(os.path.join(HERE, "taipei101_tray_ring_stand.stl"))

    preview(mesh, size, prof, z_seat)
