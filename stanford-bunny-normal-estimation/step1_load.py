"""Step 1: load the Stanford Bunny ASCII PLY and sanity-check the point cloud."""
import numpy as np
import re

def load_ascii_ply_vertices(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    n_verts = None
    header_end = None
    props = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('element vertex'):
            n_verts = int(line.split()[-1])
        elif line.startswith('property') and n_verts is not None and 'element face' not in ''.join(lines[:i]):
            props.append(line.split()[-1])
        elif line == 'end_header':
            header_end = i + 1
            break
    assert n_verts is not None and header_end is not None
    x_idx = props.index('x')
    y_idx = props.index('y')
    z_idx = props.index('z')
    pts = np.zeros((n_verts, 3), dtype=np.float64)
    for i in range(n_verts):
        vals = lines[header_end + i].split()
        pts[i] = [float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])]
    return pts, props

if __name__ == '__main__':
    pts, props = load_ascii_ply_vertices('bunny.ply')
    print('properties found:', props)
    print('point count:', len(pts))
    print('x range:', pts[:, 0].min(), pts[:, 0].max())
    print('y range:', pts[:, 1].min(), pts[:, 1].max())
    print('z range:', pts[:, 2].min(), pts[:, 2].max())
    np.save('bunny_points.npy', pts)
    print('saved bunny_points.npy')
