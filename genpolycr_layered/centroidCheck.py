import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
gdir = Path("./case1")   # directory with gchar.txt and gdata.bin
nx, ny, nz = 256, 256, 20

# For THIS file, set the real layer layout:
# Example (old, 2 layers): [10, 10]
# Example (new, 3 layers): [5, 10, 5]
layer_counts = [10,10]     # <-- CHANGE for the 3-layer file to [5, 10, 5]
# ----------------------------

gchar_path = gdir / "gchar.txt"
gdata_path = gdir / "gdata.bin"

# ---- load gchar.txt: id, x, y, z, Aphi, Atheta, Apsi ----
data = np.loadtxt(gchar_path)
gid   = data[:, 0].astype(int)
centx = data[:, 1]
centy = data[:, 2]
centz = data[:, 3]

# ---- per-layer centroid counts (by z slab) ----
bounds = np.cumsum(layer_counts)
z0 = 0
layer_gid_sets = []
layer_cent_sets = []
print("== Centroids by z-slab ==")
for L, span in enumerate(layer_counts):
    z1 = z0 + span
    m = (centz >= z0) & (centz < z1)
    gids = set(gid[m])
    layer_gid_sets.append(gids)
    # use integer rounding to avoid float artifacts; your centroids are ints-as-doubles anyway
    XYZn = np.stack([centx[m].round(6), centy[m].round(6), centz[m].round(6)], axis=1)
    # convert to tuples for exact set comparison
    cent_tuples = set(map(tuple, XYZn))
    layer_cent_sets.append(cent_tuples)

    if gids:
        print(f"Layer {L}: z=[{z0},{z1})  centroids={len(gids)}  gid_range=[{min(gids)}..{max(gids)}]")
    else:
        print(f"Layer {L}: z=[{z0},{z1})  centroids=0")
    z0 = z1

# ---- centroid duplication across layers (OLD vs NEW test) ----
print("\n== Centroid duplication across layers ==")
dup_any = False
for i in range(len(layer_counts)):
    for j in range(i+1, len(layer_counts)):
        dup_ij = layer_cent_sets[i].intersection(layer_cent_sets[j])
        print(f"Shared centroid positions Layer {i} ∩ {j}: {len(dup_ij)}")
        if len(dup_ij) > 0:
            dup_any = True

if dup_any:
    print(">>> Detected shared centroid positions across layers → this looks like the OLD relabeling case.")
else:
    print(">>> No shared centroid positions across layers → this looks like the NEW independent-seed layers.")

# ---- load omega to confirm ID disjointness per layer ----
omega = np.fromfile(gdata_path, dtype=np.int32, count=nx*ny*nz).reshape((nx, ny, nz), order='C')

print("\n== Grain ID sets per layer slab from omega (should be disjoint) ==")
z0 = 0
id_sets = []
for L, span in enumerate(layer_counts):
    z1 = z0 + span
    idsL = np.unique(omega[:, :, z0:z1])
    id_sets.append(set(idsL.tolist()))
    print(f"Layer {L} IDs: count={idsL.size}, min={idsL.min()}, max={idsL.max()}")
    z0 = z1

print("\n== ID overlaps between layers ==")
over_any = False
for i in range(len(id_sets)):
    for j in range(i+1, len(id_sets)):
        ov = id_sets[i].intersection(id_sets[j])
        print(f"Overlap IDs Layer {i} ∩ {j}: {len(ov)}")
        if len(ov) > 0:
            over_any = True

if over_any:
    print(">>> WARNING: Found overlapping IDs across layers (unexpected).")
else:
    print(">>> OK: Grain ID sets are disjoint per layer.")
