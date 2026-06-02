import pathlib
import sys

import torch


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from oat.tokenizer.oat.augment.so3_action_chunk_aug import SO3ActionChunkAug


def _make_actions(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    batch_size, horizon = 4, 8
    actions = torch.zeros(batch_size, horizon, 7, device=device, dtype=dtype)
    actions[..., 0:3] = torch.randn(batch_size, horizon, 3, device=device).to(dtype) * 0.1

    base_rot = torch.tensor(
        [
            [0.02, -0.01, 0.03],
            [-0.04, 0.02, 0.01],
            [0.01, 0.05, -0.02],
            [0.03, -0.03, 0.04],
        ],
        device=device,
        dtype=dtype,
    )
    actions[..., 3:6] = base_rot[:, None, :]
    actions[..., 6] = torch.linspace(-1.0, 1.0, batch_size, device=device).to(dtype)[:, None]
    return actions


def _assert_exactly_unchanged(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    assert torch.equal(actual, expected), f"{label} should leave actions exactly unchanged"


def _run_core_checks(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    actions = _make_actions(device=device, dtype=dtype)

    aug_disabled = SO3ActionChunkAug(p=0.0, max_angle_deg=15.0).to(device)
    aug_disabled.train()
    _assert_exactly_unchanged(aug_disabled(actions), actions, "p=0")

    aug_eval = SO3ActionChunkAug(p=1.0, max_angle_deg=15.0).to(device)
    aug_eval.eval()
    _assert_exactly_unchanged(aug_eval(actions), actions, "eval mode")

    aug = SO3ActionChunkAug(
        p=1.0,
        max_angle_deg=20.0,
        mode="left_noise",
        augment_position=False,
    ).to(device)
    aug.train()
    out = aug(actions)

    assert out.shape == actions.shape, "Augmentation must preserve shape"
    assert out.dtype == actions.dtype, "Augmentation must preserve dtype"
    assert out.device == actions.device, "Augmentation must preserve device"
    assert torch.isfinite(out.float()).all(), "Augmentation produced NaNs or infs"

    _assert_exactly_unchanged(out[..., 0:3], actions[..., 0:3], "augment_position=False")
    _assert_exactly_unchanged(out[..., 6:7], actions[..., 6:7], "gripper")

    tol = 2.0e-3 if dtype in (torch.float16, torch.bfloat16) else 1.0e-5
    rot_change = (out[..., 3:6].float() - actions[..., 3:6].float()).abs().amax(dim=(1, 2))
    assert (rot_change > tol).any(), "Expected at least one chunk rotation to change"

    rot_repeat_error = (out[:, 1:, 3:6].float() - out[:, :1, 3:6].float()).abs().max()
    assert rot_repeat_error <= tol, (
        "Repeated rotations within a chunk should remain identical after augmentation"
    )


def main() -> None:
    _run_core_checks(torch.device("cpu"), torch.float32)
    _run_core_checks(torch.device("cpu"), torch.float64)

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        _run_core_checks(cuda, torch.float32)
        _run_core_checks(cuda, torch.float16)
        if torch.cuda.is_bf16_supported():
            _run_core_checks(cuda, torch.bfloat16)

    print("SO(3) action chunk augmentation smoke test passed")


if __name__ == "__main__":
    main()
