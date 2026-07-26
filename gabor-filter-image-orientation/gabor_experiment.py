import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


# --- 1. Single Gabor kernel ---
kernel = cv2.getGaborKernel(
    ksize=(31, 31), sigma=4.0, theta=0, lambd=10.0, gamma=0.5, psi=0
)

fig = plt.figure(figsize=(5, 5))
plt.imshow(kernel, cmap="gray")
plt.title("Gabor Kernel")
plt.axis("off")
save(fig, "kernel_theta0.png")

# --- 2. Kernels at different theta ---
thetas = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, theta in zip(axes, thetas):
    k = cv2.getGaborKernel(
        ksize=(31, 31), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5, psi=0
    )
    ax.imshow(k, cmap="gray")
    ax.set_title(f"theta = {np.degrees(theta):.0f}°")
    ax.axis("off")
save(fig, "kernels_thetas.png")

# --- 3. Experiment 1: vertical vs horizontal stripes ---
image = np.zeros((256, 256), dtype=np.uint8)

for x in range(0, 128, 16):
    image[:, x:x + 4] = 255

for y in range(0, 256, 16):
    image[y:y + 4, 128:] = 255

fig = plt.figure(figsize=(6, 6))
plt.imshow(image, cmap="gray")
plt.title("Vertical vs Horizontal")
plt.axis("off")
save(fig, "exp1_input.png")

results = []
print("=== Experiment 1: vertical vs horizontal ===")
for theta in thetas:
    k = cv2.getGaborKernel(
        ksize=(31, 31), sigma=4.0, theta=theta, lambd=16.0, gamma=0.5, psi=0
    )
    filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, k)
    results.append(filtered)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(filtered), cmap="gray")
    plt.title(f"theta = {np.degrees(theta):.0f}°")
    plt.axis("off")
    save(fig, f"exp1_theta{int(np.degrees(theta))}.png")

for theta, filtered in zip(thetas, results):
    response = np.abs(filtered)
    left_score = response[:, :128].mean()
    right_score = response[:, 128:].mean()
    print(
        f"theta={np.degrees(theta):.0f}° "
        f"left(vertical-region)={left_score:.2f} "
        f"right(horizontal-region)={right_score:.2f}"
    )

# --- 4. Experiment 2: fine vs coarse stripes ---
image2 = np.zeros((256, 256), dtype=np.uint8)

for x in range(0, 128, 8):
    image2[:, x:x + 2] = 255

for x in range(128, 256, 24):
    image2[:, x:x + 4] = 255

fig = plt.figure(figsize=(6, 6))
plt.imshow(image2, cmap="gray")
plt.title("Fine vs Coarse Stripes")
plt.axis("off")
save(fig, "exp2_input.png")

print("\n=== Experiment 2: fine vs coarse stripes (theta=0) ===")
lambdas = [4, 8, 16, 32]
for lambd in lambdas:
    k = cv2.getGaborKernel(
        ksize=(31, 31), sigma=4.0, theta=0, lambd=lambd, gamma=0.5, psi=0
    )
    filtered = cv2.filter2D(image2.astype(np.float32), cv2.CV_32F, k)
    response = np.abs(filtered)

    left_score = response[:, :128].mean()
    right_score = response[:, 128:].mean()
    print(f"lambda={lambd}: fine={left_score:.2f}, coarse={right_score:.2f}")

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(response, cmap="gray")
    plt.title(f"lambda = {lambd}")
    plt.axis("off")
    save(fig, f"exp2_lambda{lambd}.png")

# --- 5. 16 Gabor feature maps grid (theta x lambda) ---
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, theta in enumerate(thetas):
    for j, lambd in enumerate(lambdas):
        k = cv2.getGaborKernel(
            ksize=(31, 31), sigma=4.0, theta=theta, lambd=lambd, gamma=0.5, psi=0
        )
        filtered = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, k)
        ax = axes[i, j]
        ax.imshow(np.abs(filtered), cmap="gray")
        ax.set_title(f"θ={np.degrees(theta):.0f}° λ={lambd}", fontsize=9)
        ax.axis("off")
save(fig, "exp3_feature_grid.png")

print("\nDone. Images written to", OUT_DIR)
