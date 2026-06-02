"""Draw a simple diagram illustrating the SO(3) action-chunk augmentation."""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expmap(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        return np.eye(3)
    k = omega / theta
    K = hat(k)
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def logmap(R):
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    return theta / (2 * math.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


# ── Build a synthetic 16-step rotvec trajectory ────────────────────────────
rng = np.random.default_rng(0)
T = 16
t = np.linspace(0, 1, T)
rotvecs = np.stack(
    [0.6 * np.sin(2 * np.pi * t), 0.3 * np.cos(np.pi * t), 0.4 * t], axis=1
)

# Sample one Q per chunk: axis ~ Uniform(S^2), angle ~ U[0, max_angle]
max_angle = math.radians(30.0)
axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
angle = math.radians(30.0)
omega = axis * angle
Q = expmap(omega)

# Apply Q · R(t) (left_noise mode)
rotvecs_aug = np.stack([logmap(Q @ expmap(w)) for w in rotvecs])


# ── Layout: pipeline (top) + 3D rotvec scatter (bottom) ────────────────────
fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[0.9, 1.5], hspace=0.30)

# === Top: pipeline diagram ===
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")

def box(x, y, w, h, text, color="#e8f0fe", edge="#1a73e8"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            linewidth=1.5, facecolor=color, edgecolor=edge,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#444",
    ))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.18, label,
                ha="center", fontsize=8, color="#333", style="italic")


# Row 1: input action chunk → decomp
box(0.2, 2.5, 2.4, 1.2,
    "raw action chunk\n$a \\in \\mathbb{R}^{B \\times 16 \\times 7}$\n[pos(3) | rotvec(3) | grip(1)]",
    color="#fff4e5", edge="#e67e22")
arrow(2.6, 3.1, 3.4, 3.1, "slice [3:6]")
box(3.4, 2.5, 2.0, 1.2,
    "rotvec(t)\n$\\omega(t) \\in \\mathbb{R}^3$\nfor $t=0..15$",
    color="#e8f0fe", edge="#1a73e8")
arrow(5.4, 3.1, 6.2, 3.1, "$R(t)=\\exp(\\omega(t))$")
box(6.2, 2.5, 2.0, 1.2,
    "$R(t) \\in SO(3)$\n(rotation matrices)",
    color="#e8f0fe", edge="#1a73e8")

# Right side: sampled Q
box(9.3, 2.5, 3.4, 1.2,
    "sample one $Q \\in SO(3)$ per chunk\n"
    "axis $n \\sim \\mathcal{U}(S^2)$,  "
    "angle $\\theta \\sim \\mathcal{U}(0, 30°)$\n"
    "$Q = \\exp(\\theta \\cdot n)$",
    color="#fde2e2", edge="#c0392b")

# Row 2: combine
arrow(7.2, 2.5, 7.2, 1.7)
arrow(11.0, 2.5, 11.0, 1.7)
box(6.0, 0.5, 6.2, 1.2,
    "left_noise:   $R_{\\rm aug}(t) = Q \\cdot R(t)$\n"
    "(same $Q$ shared across all 16 steps)",
    color="#e3f7e1", edge="#27ae60")

# Output column
arrow(8.0, 0.5, 8.0, 0.1)
arrow(8.0, 0.1, 3.4, 0.1)
arrow(3.4, 0.1, 3.4, 0.5)
box(2.4, 0.5, 3.0, 1.2,
    "$\\omega_{\\rm aug}(t) = \\log(R_{\\rm aug}(t))$\n"
    "replace rotvec slice\n(pos & grip untouched)",
    color="#e8f0fe", edge="#1a73e8")
arrow(2.4, 1.1, 1.6, 1.1)
box(-0.3, 0.5, 1.9, 1.2,
    "augmented\nchunk\n$\\to$ normalize $\\to$ encoder",
    color="#fff4e5", edge="#e67e22")

ax.set_title(
    "SO(3) action-chunk augmentation pipeline  (SO3ActionChunkAug, mode=left_noise)",
    fontsize=11, pad=4,
)

# === Bottom: 3D visualization of the rotvec trajectory before/after ========
ax3d = fig.add_subplot(gs[1], projection="3d")
ax3d.plot(rotvecs[:, 0], rotvecs[:, 1], rotvecs[:, 2],
          "o-", color="#1a73e8", label="original $\\omega(t)$", markersize=4)
ax3d.plot(rotvecs_aug[:, 0], rotvecs_aug[:, 1], rotvecs_aug[:, 2],
          "s-", color="#c0392b",
          label=f"augmented $\\omega_{{\\rm aug}}(t)$  "
                f"($\\theta = {math.degrees(angle):.1f}°$)",
          markersize=4)

# Draw Q's axis as a dashed line through origin
L = 0.8
ax3d.plot([-L * axis[0], L * axis[0]],
          [-L * axis[1], L * axis[1]],
          [-L * axis[2], L * axis[2]],
          "--", color="#666", linewidth=1.0, label="Q axis $n$")

ax3d.set_xlabel("$\\omega_x$", labelpad=18)
ax3d.set_ylabel("$\\omega_y$", labelpad=18)
ax3d.set_zlabel("$\\omega_z$", labelpad=14)
ax3d.tick_params(axis="x", pad=4)
ax3d.tick_params(axis="y", pad=4)
ax3d.tick_params(axis="z", pad=6)
ax3d.legend(loc="center left", fontsize=9, bbox_to_anchor=(1.05, 0.5),
            frameon=True, framealpha=0.9)
ax3d.set_title(
    "Effect on the rotvec trajectory: the whole 16-step path is rigidly "
    "rotated by $Q$\n(one perturbation per chunk, shared across all timesteps)",
    fontsize=10, pad=24,
)
ax3d.view_init(elev=22, azim=-55)

plt.subplots_adjust(left=0.03, right=0.85, top=0.95, bottom=0.05, hspace=0.45)
out = "/workspace/oat/scripts/so3_aug_diagram.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved diagram to {out}")
