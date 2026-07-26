import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, transform
import os

plt.rcParams["font.family"] = "Meiryo"

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_stripe(size, angle_deg, wavelength):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    theta = np.deg2rad(angle_deg)
    xr = x * np.cos(theta) + y * np.sin(theta)
    return np.sin(2 * np.pi * xr / wavelength)


def spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return np.log(1 + np.abs(fshift))


# --- 1. Synthetic stripes: orientation/frequency -> spectrum position ---
size = 256
patterns = [
    ("垂直・細かい", make_stripe(size, 0, 8)),
    ("垂直・粗い", make_stripe(size, 0, 32)),
    ("斜め45度", make_stripe(size, 45, 16)),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (label, img) in enumerate(patterns):
    axes[0, i].imshow(img, cmap="gray")
    axes[0, i].set_title(label)
    axes[0, i].axis("off")

    mag = spectrum(img)
    axes[1, i].imshow(mag, cmap="gray")
    axes[1, i].set_title("周波数スペクトル")
    axes[1, i].axis("off")
save(fig, "1_stripes_and_spectra.png")
print("1 done")

# --- 2. Real photo + its spectrum ---
camera = data.camera().astype(np.float64)
mag_camera = spectrum(camera)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(camera, cmap="gray")
axes[0].set_title("元画像")
axes[0].axis("off")
axes[1].imshow(mag_camera, cmap="gray")
axes[1].set_title("周波数スペクトル")
axes[1].axis("off")
save(fig, "2_photo_spectrum.png")
print("2 done")

# --- 3. Low-pass / high-pass reconstruction ---
f = np.fft.fft2(camera)
fshift = np.fft.fftshift(f)

rows, cols = camera.shape
crow, ccol = rows // 2, cols // 2
Y, X = np.ogrid[:rows, :cols]
dist = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)

radius = 30
low_mask = dist <= radius
high_mask = ~low_mask

low_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * low_mask)))
high_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * high_mask)))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(camera, cmap="gray")
axes[0].set_title("元画像")
axes[0].axis("off")
axes[1].imshow(low_img, cmap="gray")
axes[1].set_title(f"低周波のみ（半径{radius}px以内）")
axes[1].axis("off")
axes[2].imshow(high_img, cmap="gray")
axes[2].set_title("高周波のみ")
axes[2].axis("off")
save(fig, "3_lowpass_highpass.png")
print("3 done")

# --- 4/5. Amplitude / phase swap between two real photos ---
photo_a = data.camera().astype(np.float64)
photo_b = transform.resize(
    color.rgb2gray(data.astronaut()), photo_a.shape, anti_aliasing=True
) * 255

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(photo_a, cmap="gray")
axes[0].set_title("画像A")
axes[0].axis("off")
axes[1].imshow(photo_b, cmap="gray")
axes[1].set_title("画像B")
axes[1].axis("off")
save(fig, "4_ab_photos.png")
print("4 done")

Fa = np.fft.fft2(photo_a)
Fb = np.fft.fft2(photo_b)

amp_a, phase_a = np.abs(Fa), np.angle(Fa)
amp_b, phase_b = np.abs(Fb), np.angle(Fb)

mix1 = amp_a * np.exp(1j * phase_b)  # A's amplitude + B's phase
mix2 = amp_b * np.exp(1j * phase_a)  # B's amplitude + A's phase

img_mix1 = np.abs(np.fft.ifft2(mix1))
img_mix2 = np.abs(np.fft.ifft2(mix2))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_mix1, cmap="gray")
axes[0].set_title("Aの振幅 + Bの位相")
axes[0].axis("off")
axes[1].imshow(img_mix2, cmap="gray")
axes[1].set_title("Bの振幅 + Aの位相")
axes[1].axis("off")
save(fig, "5_phase_amplitude_swap.png")
print("5 done")

print("Done. Images written to", OUT_DIR)
