"""Standalone test for the novel-view eval harness (eval_novel_view.py).

No training and no checkpoint required: a random policy exercises the full
multi-viewpoint rollout loop. Verifies
  1. moving the camera changes the rendered agentview image at reset,
  2. restoring the identity viewpoint reproduces the base render exactly,
  3. a sweep produces per-viewpoint metrics, success flags, and videos.

Run from the repo root:
    python tests/test_novel_view_harness.py

Requires a robomimic image dataset at data/robomimic/datasets/lift/ph/image.hdf5
(see PROGRESS.md for the download command).

NOTE: checks 1-2 build a *rendering* env in this process and check 3 forks
workers that each create their own rendering env. Creating a GL context in the
parent before forking deadlocks the workers, so checks 1-2 run in a separate
subprocess (--render-check). Do not merge them into one process.
"""
import os
import sys

# repo convention: tests run from the repo root
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)

import shutil
import subprocess
import tempfile

import numpy as np
import torch

from eval_novel_view import (
    PRESETS, ViewpointImageWrapper, run_novel_view_eval, to_json_log)
from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
import robomimic.utils.file_utils as FileUtils


DATASET_PATH = 'data/robomimic/datasets/lift/ph/image.hdf5'
SHAPE_META = {
    'obs': {
        'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'robot0_eef_pos': {'shape': [3]},
        'robot0_eef_quat': {'shape': [4]},
        'robot0_gripper_qpos': {'shape': [2]},
    },
    'action': {'shape': [7]},
}


class RandomPolicy(BaseImagePolicy):
    """Minimal policy stub: uniformly random actions in [-1, 1]."""

    def __init__(self, action_dim=7, n_action_steps=8):
        super().__init__()
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps

    def predict_action(self, obs_dict):
        some_obs = next(iter(obs_dict.values()))
        B = some_obs.shape[0]
        return {'action': torch.rand(
            (B, self.n_action_steps, self.action_dim),
            device=some_obs.device, dtype=some_obs.dtype) * 2 - 1}

    def set_normalizer(self, normalizer):
        pass


def _build_env():
    env_meta = FileUtils.get_env_metadata_from_dataset(DATASET_PATH)
    env_meta['env_kwargs']['use_object_obs'] = False
    robomimic_env = create_env(env_meta=env_meta, shape_meta=SHAPE_META)
    robomimic_env.env.hard_reset = False
    return ViewpointImageWrapper(
        env=robomimic_env,
        shape_meta=SHAPE_META,
        init_state=None,
        render_obs_key='agentview_image')


def render_check():
    """Checks 1+2, in their own process (see module docstring)."""
    wrapper = _build_env()
    sim = wrapper.env.env.sim
    cid = sim.model.camera_name2id('agentview')
    base_pos = sim.model.cam_pos[cid].copy()

    wrapper.set_viewpoint({'camera': 'agentview', 'name': 'az_0'})
    wrapper.seed(0)
    img_base = wrapper.reset()['agentview_image'].copy()

    wrapper.set_viewpoint(
        {'camera': 'agentview', 'name': 'az_p30', 'azimuth_deg': 30})
    wrapper.seed(0)
    img_az30 = wrapper.reset()['agentview_image'].copy()
    moved_pos = sim.model.cam_pos[cid].copy()

    assert not np.allclose(moved_pos, base_pos), 'camera did not move'
    # 30 deg orbit about z: radius in the xy plane is preserved
    assert np.isclose(np.linalg.norm(moved_pos[:2]), np.linalg.norm(base_pos[:2]),
                      atol=1e-5), 'azimuth orbit changed the radius'
    assert np.abs(img_base - img_az30).mean() > 0.01, 'render did not change'

    wrapper.set_viewpoint({'camera': 'agentview', 'name': 'az_0'})
    wrapper.seed(0)
    img_restored = wrapper.reset()['agentview_image'].copy()
    assert np.allclose(sim.model.cam_pos[cid], base_pos), 'pose not restored'
    assert np.abs(img_base - img_restored).max() < 1e-6, \
        'identity viewpoint does not reproduce the base render'
    print('1+2 OK: camera move changes render; identity restores exactly')


def test():
    assert os.path.exists(DATASET_PATH), f"missing dataset: {DATASET_PATH}"

    # checks 1+2 in a subprocess (GL context must not exist before forking)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--render-check'],
        cwd=repo_root)
    assert proc.returncode == 0, 'render check subprocess failed'

    # ---- 3: full multi-viewpoint rollout with a random policy
    out_dir = tempfile.mkdtemp(prefix='novel_view_test_')
    try:
        preset = PRESETS['azimuth_sweep3']
        policy = RandomPolicy(action_dim=7, n_action_steps=8)
        policy.to('cpu')
        log_data = run_novel_view_eval(
            policy=policy,
            output_dir=out_dir,
            shape_meta=SHAPE_META,
            dataset_path=DATASET_PATH,
            preset=preset,
            n_test=2,
            n_test_vis=1,
            n_envs=2,
            n_obs_steps=2,
            n_action_steps=8,
            max_steps=40,
            test_start_seed=0)
        json_log = to_json_log(log_data)

        for vp in preset['viewpoints']:
            prefix = f"test/{vp['name']}/"
            assert prefix + 'mean_score' in json_log, f'missing {prefix}mean_score'
            assert prefix + 'success_rate' in json_log, f'missing {prefix}success_rate'
            rate = json_log[prefix + 'success_rate']
            assert 0.0 <= rate <= 1.0, f'success_rate out of range: {rate}'
            for seed in (0, 1):
                assert prefix + f'success_{seed}' in json_log
                assert prefix + f'sim_max_reward_{seed}' in json_log
            # one video per viewpoint (n_test_vis=1)
            video_dir = os.path.join(out_dir, 'media', vp['name'])
            videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            assert len(videos) == 1, f'expected 1 video in {video_dir}, got {videos}'
            assert json_log[prefix + 'sim_video_0'].startswith(video_dir), \
                'video path not logged'
        print('3 OK: per-viewpoint metrics, success flags, and videos produced')
        print('sample log keys:', sorted(
            k for k in json_log
            if k.endswith('mean_score') or k.endswith('success_rate')))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    print('ALL NOVEL-VIEW HARNESS CHECKS PASSED')


if __name__ == '__main__':
    if '--render-check' in sys.argv:
        render_check()
    else:
        test()
