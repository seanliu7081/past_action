"""Illustrate raw past actions vs. acceleration vs. jerk as a 3-panel figure."""
import numpy as np
import matplotlib.pyplot as plt


def arrow_at_end(ax, x, y, color):
    """Draw a small arrow at the last sample of (x, y) pointing in the curve's
    instantaneous direction so the curve clearly has a 'head'."""
    dx, dy = x[-1] - x[-2], y[-1] - y[-2]
    ax.annotate(
        "",
        xy=(x[-1] + dx * 0.6, y[-1] + dy * 0.6),
        xytext=(x[-1], y[-1]),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2),
    )


# Dense sampling for nice curves
t = np.linspace(0, 6, 500)

# ── Raw past action: smooth, low-frequency ─────────────────────────────────
raw = 0.6 * np.sin(0.7 * t) + 0.3 * np.cos(0.4 * t + 0.5)

# ── Acceleration: moderate-frequency, larger swings ────────────────────────
acc = (
    0.6 * np.sin(2.0 * t + 0.3)
    + 0.3 * np.sin(3.5 * t - 0.4)
)

# ── Jerk: jerky, high-frequency, sharp spikes ──────────────────────────────
rng = np.random.default_rng(7)
jerk_base = (
    0.5 * np.sin(8.0 * t)
    + 0.4 * np.sin(13.0 * t + 0.7)
    + 0.3 * np.sin(19.0 * t - 0.2)
)
# Add a few sharp impulse-like kinks
spike_centers = [1.0, 2.4, 3.6, 4.9]
spike = np.zeros_like(t)
for c in spike_centers:
    spike += 0.6 * np.sign(np.sin(40 * (t - c))) * np.exp(-((t - c) ** 2) / 0.04)
jerk = jerk_base + spike

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                         facecolor="white")

curves = [
    (raw,  "#1a73e8", "raw past action  $a_{t-k}$",
     "smooth, low-frequency — the command trajectory itself"),
    (acc,  "#27ae60", "acceleration  $\\Delta a = a_{t-1} - a_{t-2}$",
     "moderate-frequency — first-order changes between commands"),
    (jerk, "#e67e22", "jerk  $\\Delta^2 a = a_{t-1} - 2 a_{t-2} + a_{t-3}$",
     "high-frequency, jerky — second-order changes"),
]

for ax, (y, color, label, subtitle) in zip(axes, curves):
    ax.plot(t, y, color=color, linewidth=2.2)
    arrow_at_end(ax, t, y, color)
    ax.set_xlim(t[0] - 0.2, t[-1] + 0.6)
    ymax = max(abs(y.min()), abs(y.max())) * 1.25
    ax.set_ylim(-ymax, ymax)
    ax.set_facecolor("white")
    ax.axis("off")

plt.tight_layout()
out = "/workspace/oat/scripts/enriched_past_curves.png"
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved diagram to {out}")
