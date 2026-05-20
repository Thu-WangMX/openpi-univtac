"""Convert UniVTAC HDF5 demonstrations to LeRobot format for OpenPI training.

Example:
    uv run examples/univtac/convert_univtac_data_to_lerobot.py \
        --raw-dir /vepfs-C区/visuotactile/UniVTAC/data/lift_bottle_official/lift_bottle/clean \
        --repo-id local/univtac_lift_bottle \
        --episodes 10
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

_TASK_PROMPTS = {
    "lift_bottle": "lift the bottle",
    "grasp_classify": "grasp and classify the object",
    "insert_HDMI": "insert the HDMI connector",
    "insert_tube": "insert the tube",
    "insert_hole": "insert the peg into the hole",
    "pull_out_key": "pull out the key",
    "lift_can": "lift the can",
    "put_bottle_in_shelf": "put the bottle in the shelf",
}


@dataclasses.dataclass(frozen=True)
class ConvertConfig:
    raw_dir: Path
    repo_id: str = "local/univtac_lift_bottle"
    root: Path | None = None
    task: str | None = None
    task_name: str | None = None
    episodes: int | None = None
    fps: int = 10
    downsample: int = 1
    action_horizon: int = 32
    mode: Literal["vision", "visuotactile"] = "visuotactile"
    state_mode: Literal["joint", "extended"] = "joint"
    action_source: Literal["next_joint", "joint_action"] = "next_joint"
    use_wrist: bool = False
    use_videos: bool = False
    image_writer_processes: int = 0
    image_writer_threads: int = 0
    overwrite: bool = True


def _sorted_hdf5_files(raw_dir: Path) -> list[Path]:
    hdf5_dir = raw_dir / "hdf5"
    if hdf5_dir.exists():
        raw_dir = hdf5_dir
    files = sorted(raw_dir.glob("*.hdf5"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under {raw_dir}")
    return files


def _infer_task_name(raw_dir: Path) -> str:
    parts = raw_dir.resolve().parts
    for name in _TASK_PROMPTS:
        if name in parts:
            return name
    return raw_dir.parent.name if raw_dir.name == "clean" else raw_dir.name


def _decode_jpeg(buf: bytes | np.bytes_) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG image from UniVTAC HDF5 buffer")
    # HDF5 frames were written through OpenCV. Convert them back to standard
    # display RGB so training matches PIL/matplotlib visualization semantics.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _first_existing(group: h5py.File, keys: list[str]) -> str | None:
    return next((key for key in keys if key in group), None)


def _features(*, mode: str, use_wrist: bool, state_dim: int) -> dict:
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (270, 480, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["actions"],
        },
    }
    if use_wrist:
        features["observation.wrist_image"] = {
            "dtype": "image",
            "shape": (270, 480, 3),
            "names": ["height", "width", "channels"],
        }
    if mode == "visuotactile":
        features["observation.left_tactile_image"] = {
            "dtype": "image",
            "shape": (240, 320, 3),
            "names": ["height", "width", "channels"],
        }
        features["observation.right_tactile_image"] = {
            "dtype": "image",
            "shape": (240, 320, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _create_dataset(config: ConvertConfig) -> LeRobotDataset:
    root = config.root if config.root is not None else HF_LEROBOT_HOME / config.repo_id
    if root.exists():
        if not config.overwrite:
            raise FileExistsError(f"Output LeRobot dataset already exists: {root}")
        shutil.rmtree(root)

    return LeRobotDataset.create(
        repo_id=config.repo_id,
        root=root,
        fps=config.fps,
        robot_type="univtac_franka",
        features=_features(
            mode=config.mode,
            use_wrist=config.use_wrist,
            state_dim=29 if config.state_mode == "extended" else 8,
        ),
        use_videos=config.use_videos,
        image_writer_processes=config.image_writer_processes,
        image_writer_threads=config.image_writer_threads,
    )


def _episode_frame_indices(length: int, downsample: int) -> np.ndarray:
    if length < 2:
        return np.array([], dtype=np.int64)
    return np.arange(0, length - 1, downsample, dtype=np.int64)


def _tactile_prefix(ep: h5py.File, side: Literal["left", "right"]) -> str | None:
    return _first_existing(ep, [f"tactile/{side}_tactile", f"tactile/{side}_gsmini"])


def _build_state(ep: h5py.File, joints: np.ndarray, *, config: ConvertConfig, ep_path: Path) -> np.ndarray:
    joint_state = joints[:, :8]
    if config.state_mode == "joint":
        return joint_state

    if "embodiment/ee" not in ep:
        raise KeyError(f"--state-mode=extended requested, but embodiment/ee is missing in {ep_path}")

    left_prefix = _tactile_prefix(ep, "left")
    right_prefix = _tactile_prefix(ep, "right")
    if left_prefix is None or right_prefix is None:
        raise KeyError(f"--state-mode=extended requested, but tactile groups are missing in {ep_path}")

    left_pose_key = f"{left_prefix}/pose"
    right_pose_key = f"{right_prefix}/pose"
    if left_pose_key not in ep or right_pose_key not in ep:
        raise KeyError(f"--state-mode=extended requested, but tactile pose is missing in {ep_path}")

    return np.concatenate(
        [
            joint_state,
            ep["embodiment/ee"][:].astype(np.float32),
            ep[left_pose_key][:].astype(np.float32),
            ep[right_pose_key][:].astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)


def _add_episode(dataset: LeRobotDataset, ep_path: Path, *, config: ConvertConfig, prompt: str) -> int:
    with h5py.File(ep_path, "r") as ep:
        joints = ep["embodiment/joint"][:].astype(np.float32)
        if joints.shape[-1] < 8:
            raise ValueError(f"Expected at least 8 joint dims in {ep_path}, got {joints.shape}")
        state = _build_state(ep, joints, config=config, ep_path=ep_path)
        if config.action_source == "joint_action" and "embodiment/joint_action" in ep:
            action = ep["embodiment/joint_action"][:].astype(np.float32)[:, :8]
        else:
            action = joints[:, :8]

        head_key = _first_existing(ep, ["observation/head/rgb"])
        if head_key is None:
            raise KeyError(f"No head RGB observation found in {ep_path}")

        wrist_key = _first_existing(ep, ["observation/wrist/rgb"])
        if config.use_wrist and wrist_key is None:
            raise KeyError(f"--use-wrist requested, but no wrist RGB observation found in {ep_path}")

        left_tactile_key = _first_existing(ep, ["tactile/left_tactile/rgb_marker", "tactile/left_gsmini/rgb_marker"])
        right_tactile_key = _first_existing(ep, ["tactile/right_tactile/rgb_marker", "tactile/right_gsmini/rgb_marker"])
        if config.mode == "visuotactile" and (left_tactile_key is None or right_tactile_key is None):
            raise KeyError(f"Visuotactile mode requested, but tactile RGB marker streams are missing in {ep_path}")

        frame_indices = _episode_frame_indices(len(state), config.downsample)
        for i in frame_indices:
            frame = {
                "observation.image": _decode_jpeg(ep[head_key][i]),
                "observation.state": state[i],
                "actions": action[i + 1],
                "task": prompt,
            }
            if config.use_wrist:
                frame["observation.wrist_image"] = _decode_jpeg(ep[wrist_key][i])
            if config.mode == "visuotactile":
                frame["observation.left_tactile_image"] = _decode_jpeg(ep[left_tactile_key][i])
                frame["observation.right_tactile_image"] = _decode_jpeg(ep[right_tactile_key][i])
            dataset.add_frame(frame)

    if len(frame_indices) > 0:
        dataset.save_episode()
    return len(frame_indices)


def main(config: ConvertConfig) -> None:
    files = _sorted_hdf5_files(config.raw_dir)
    if config.episodes is not None:
        files = files[: config.episodes]

    task_name = config.task_name or _infer_task_name(config.raw_dir)
    prompt = config.task or _TASK_PROMPTS.get(task_name, task_name.replace("_", " "))

    dataset = _create_dataset(config)
    total_frames = 0
    for ep_path in tqdm.tqdm(files, desc="Converting UniVTAC episodes"):
        total_frames += _add_episode(dataset, ep_path, config=config, prompt=prompt)

    print(f"Wrote {len(files)} episodes / {total_frames} frames")
    print(f"Repo id: {config.repo_id}")
    print(f"Root: {dataset.root}")
    print(f"Prompt: {prompt!r}")


if __name__ == "__main__":
    tyro.cli(main)
