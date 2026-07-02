"""
trajectory_feasibility.py — feasibility / legality metrics for LIBERO OSC_POSE
rollouts, complementing trajectory_smoothness.py.

    L0 command legality  -> needs only the executed action stream
    L1 joint feasibility -> needs joint_pos / joint_vel   (present in sim obs)
    L1 singularity       -> needs the geometric Jacobian  (optional)
    L3 realized motion    -> needs eef_pos / eef_quat      (present in sim obs)

VERIFY before trusting:
  * action bounds low/high -> LIBERO OSC_POSE input range, default [-1, 1].
  * PANDA_QPOS_*/QVEL_LIM  -> confirm against the exact robot XML in this build.
"""

import numpy as np

# Franka Emika Panda (7-DoF). VERIFY against the robot XML used by this LIBERO build.
PANDA_QPOS_LOW  = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_QPOS_HIGH = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
PANDA_QVEL_LIM  = np.array([ 2.1750,  2.1750,  2.1750,  2.1750,  2.6100,  2.6100,  2.6100])


def _fd(x, n):
    for _ in range(n):
        x = np.diff(x, axis=0)
    return x


def sparc(speed, fs, fc=10.0, amp_th=0.05, pad_pow=4):
    """Spectral Arc Length of a scalar speed profile (Balasubramanian et al.).
    <= 0; closer to 0 = smoother. Same conventions as the smoothness module."""
    v = np.asarray(speed, np.float64)
    if v.size < 2 or np.allclose(v, 0.0):
        return 0.0
    N = int(2 ** (np.ceil(np.log2(len(v))) + pad_pow))
    Vf = np.abs(np.fft.rfft(v, n=N))
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    Vf = Vf / (Vf.max() + 1e-12)
    keep = f <= fc
    f, Vf = f[keep], Vf[keep]
    inband = np.where(Vf >= amp_th)[0]
    if inband.size < 2:
        return 0.0
    last = inband[-1]
    f, Vf = f[: last + 1], Vf[: last + 1]
    fn = (f - f[0]) / (f[-1] - f[0] + 1e-12)
    return float(-np.sum(np.sqrt(np.diff(fn) ** 2 + np.diff(Vf) ** 2)))


def command_legality(actions, low=-1.0, high=1.0, sat_th=0.99,
                     pos_slice=(0, 3), rot_slice=(3, 6), grip_idx=6):
    """Did the policy emit actions the action space can express? Measure on the
    PRE-CLIP action; if the stream is already clipped, OOB collapses to 0."""
    a = np.asarray(actions, np.float64)
    ps, rs = slice(*pos_slice), slice(*rot_slice)
    excess = np.maximum.reduce([a - high, low - a, np.zeros_like(a)])
    oob = excess > 0
    sat = np.abs(a) > sat_th
    g = a[:, grip_idx]
    flips = np.abs(np.diff(np.sign(g))) > 1.0
    return {
        "oob_frac_pos":  float(oob[:, ps].any(-1).mean()),
        "oob_frac_rot":  float(oob[:, rs].any(-1).mean()),
        "oob_frac_grip": float(oob[:, grip_idx].mean()),
        "oob_frac_any":  float(oob.any(-1).mean()),
        "oob_excess_max": float(excess.max()),
        "sat_frac_pos":  float(sat[:, ps].any(-1).mean()),
        "sat_frac_rot":  float(sat[:, rs].any(-1).mean()),
        "grip_flip_rate": float(flips.mean()) if flips.size else 0.0,
    }


def joint_feasibility(qpos, qvel, low=PANDA_QPOS_LOW, high=PANDA_QPOS_HIGH,
                      vlim=PANDA_QVEL_LIM, margin=0.0):
    qpos = np.asarray(qpos, np.float64)
    qvel = np.asarray(qvel, np.float64)
    lo, hi = low + margin, high - margin
    pos_viol = (qpos < lo) | (qpos > hi)
    dist = np.minimum(qpos - low[None], high[None] - qpos)
    vratio = np.abs(qvel) / vlim[None]
    return {
        "jpos_viol_frac":  float(pos_viol.any(-1).mean()),
        "jvel_viol_frac":  float((vratio > 1.0).any(-1).mean()),
        "jpos_margin_min": float(dist.min()),
        "jvel_max_ratio":  float(vratio.max()),
    }


def singularity_metrics(jacobians, manip_th=0.02):
    """Yoshikawa manipulability sqrt(det(J J^T)) and condition number."""
    J = np.asarray(jacobians, np.float64)  # [T, 6, N]
    manip, cond = [], []
    for Jt in J:
        JJt = Jt @ Jt.T
        manip.append(float(np.sqrt(max(np.linalg.det(JJt), 0.0))))
        s = np.linalg.svd(Jt, compute_uv=False)
        cond.append(float(s[0] / (s[-1] + 1e-12)))
    manip, cond = np.asarray(manip), np.asarray(cond)
    return {
        "manip_min":      float(manip.min()),
        "cond_max":       float(cond.max()),
        "near_sing_frac": float((manip < manip_th).mean()),
    }


def _quat_geodesic_speed(quat, dt):
    """Angular speed from unit quats, double-cover-safe. quat = (x,y,z,w)."""
    q = np.asarray(quat, np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    dots = np.abs(np.sum(q[1:] * q[:-1], axis=-1)).clip(-1.0, 1.0)
    return (2.0 * np.arccos(dots)) / dt


def realized_smoothness(eef_pos=None, eef_quat=None, fs=20.0):
    """vel/acc/jerk/SPARC on the realized EE trajectory + worst-case jerk tails.
    eef_pos in metres -> SI units. Rotation via geodesic angular speed."""
    dt, out = 1.0 / fs, {}
    if eef_pos is not None and len(eef_pos) >= 2:
        x = np.asarray(eef_pos, np.float64)
        v = np.linalg.norm(_fd(x, 1), axis=-1) / dt
        out["rvel_pos"], out["rsparc_pos"] = float(v.mean()), sparc(v, fs)
        if len(x) >= 3:
            out["racc_pos"] = float((np.linalg.norm(_fd(x, 2), -1) / dt ** 2).mean())
        if len(x) >= 4:
            j = np.linalg.norm(_fd(x, 3), -1) / dt ** 3
            out["rjerk_pos"] = float(j.mean())
            out["rjerk_pos_p95"], out["rjerk_pos_max"] = float(np.percentile(j, 95)), float(j.max())
    if eef_quat is not None and len(eef_quat) >= 2:
        w = _quat_geodesic_speed(eef_quat, dt)
        out["rvel_rot"], out["rsparc_rot"] = float(w.mean()), sparc(w, fs)
        if w.size >= 2:
            out["racc_rot"] = float((np.abs(_fd(w, 1)) / dt).mean())
        if w.size >= 3:
            j = np.abs(_fd(w, 2)) / dt ** 2
            out["rjerk_rot"] = float(j.mean())
            out["rjerk_rot_p95"], out["rjerk_rot_max"] = float(np.percentile(j, 95)), float(j.max())
    return out


def compute_feasibility_metrics(ep, fs=20.0, low=-1.0, high=1.0,
                                pos_slice=(0, 3), rot_slice=(3, 6), grip_idx=6):
    """`ep` may contain any subset of: action[T,D], qpos[T,7], qvel[T,7],
    eef_pos[T,3], eef_quat[T,4], jacobian[T,6,N]. Missing groups are skipped."""
    m = {}
    a = ep.get("action")
    if a is not None and len(a) >= 2:
        m.update(command_legality(a, low, high, pos_slice=pos_slice,
                                  rot_slice=rot_slice, grip_idx=grip_idx))
    qpos, qvel = ep.get("qpos"), ep.get("qvel")
    if qpos is not None and qvel is not None and len(qpos) >= 1:
        m.update(joint_feasibility(qpos, qvel))
    if ep.get("eef_pos") is not None or ep.get("eef_quat") is not None:
        m.update(realized_smoothness(ep.get("eef_pos"), ep.get("eef_quat"), fs))
    jac = ep.get("jacobian")
    if jac is not None and len(jac) >= 1:
        m.update(singularity_metrics(jac))
    m["n_steps"] = int(len(a)) if a is not None else 0
    return m


# Averaged over scored episodes (mean == overall violation rate for the frac keys)
FEAS_KEYS_MEAN = [
    "oob_frac_pos", "oob_frac_rot", "oob_frac_grip", "oob_frac_any",
    "sat_frac_pos", "sat_frac_rot", "grip_flip_rate",
    "jpos_viol_frac", "jvel_viol_frac",
    "rvel_pos", "racc_pos", "rjerk_pos", "rsparc_pos", "rjerk_pos_p95",
    "rvel_rot", "racc_rot", "rjerk_rot", "rsparc_rot", "rjerk_rot_p95",
    "near_sing_frac", "cond_max",
]
# Reduced with min/max over ALL scored episodes (true worst-case across the eval)
FEAS_KEYS_WORST = {
    "jpos_margin_min": "min", "manip_min": "min",
    "oob_excess_max": "max", "jvel_max_ratio": "max",
    "rjerk_pos_max": "max", "rjerk_rot_max": "max",
}
