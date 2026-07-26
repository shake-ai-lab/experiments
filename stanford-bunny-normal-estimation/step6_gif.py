"""Step 6: GIF of the point cloud recolored as k sweeps from too-small to too-large,
with a longer hold on the first frame so the starting (noisy) state registers."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Meiryo'
import imageio.v2 as imageio
import io
from PIL import Image

pts = np.load('bunny_points.npy')
data = np.load('ksweep_results.npz')
ks = data['ks'].tolist()

def to_rgb(normals):
    return np.clip((normals + 1) / 2, 0, 1)

# use ALL points (a 3000-pt subsample let the far side of the bunny show through
# the gaps, which made every frame look speckled regardless of k and hid the
# smoothing progression the GIF is supposed to show)
frames = []
for k in ks:
    normals_k = data[f'normals_k{k}']
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=90)
    ax.scatter(pts[:, 0], pts[:, 1], c=to_rgb(normals_k), s=3)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if k <= 8:
        tag = '(小さすぎ・ノイズ)'
    elif k >= 90:
        tag = '(大きすぎ・耳が消える)'
    else:
        tag = ''
    ax.set_title(f'k = {k}  {tag}', fontsize=13)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=90)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert('RGB').quantize(colors=128, method=Image.MEDIANCUT)
    frames.append(im)

# hold on the first frame (small k, noisy) so viewers register the starting state
gif_frames = [frames[0]] + frames
gif_durations = [900, 900] + [350] * (len(frames) - 2) + [1200]

gif_frames[0].save('ksweep.gif', save_all=True, append_images=gif_frames[1:],
                    duration=gif_durations, loop=0, optimize=True, disposal=2)
print('saved ksweep.gif, frames:', len(gif_frames))
