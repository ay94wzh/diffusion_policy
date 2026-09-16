"""
Novel-view evaluation for Diffusion Policy baselines (robomimic).

Standalone harness -- does NOT modify the diffusion_policy package. Loads a
checkpoint exactly like eval.py and builds the same robomimic env chain as
RobomimicImageRunner, but perturbs a fixed camera's pose at the mujoco_py
model level before each rollout, sweeping a set of viewpoints. The same
seeds are used across viewpoints (paired evaluation). Per-viewpoint
success rate, mean score, and videos are written to eval_log.json.

Camera pose mechanics (verified against robosuite 1.2.0 / mujoco_py 2.1.2):
- robosuite has no CameraMover before 1.3; offscreen rendering reads mjModel
  cam fields (cam_pos/cam_quat) on every render call, so mutating them
  (plus sim.forward()) moves the rendered camera.
- cam_quat is wxyz (mujoco); robosuite.utils.transform_utils is xyzw --
  convert explicitly.
- Only fixed (world) cameras can be perturbed: 'agentview' is fixed;
  'robot0_eye_in_hand' is body-attached to the gripper (do not perturb).

Usage:
python eval_novel_view.py -c <checkpoint.ckpt> -o <output_dir> -d cuda:0 --preset azimuth_sweep5
python eval_novel_view.py -c <ckpt> -o <out> --preset azimuth_sweep3 --n-test 4 --n-test-vis 2 --n-envs 4
"""

import os
import sys
import math
import copy
import pathlib
import collections
import json

# bootstrap: make repo root importable regardless of cwd
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import click
import hydra
import torch
import dill
import numpy as np
import wandb
import tqdm
import wandb.sdk.data_types.video as wv

from diffusion_policy.env_runner.robomimic_image_runner import create_env
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from robosuite.utils import transform_utils as T
import robomimic.utils.file_utils as FileUtils

# ---------------------------------------------------------------------------
# viewpoint presets
#
# All viewpoints are computed from the base (dataset) camera pose at env
# reset; the same test seeds are used across viewpoints (paired evaluation).
# azimuth: orbit about world z through the origin (robosuite tables are near
# the origin) + look-at the base camera's look-at point.
# ---------------------------------------------------------------------------

PRESETS = {
    'azimuth_sweep5': {
        'camera': 'agentview',
        'viewpoints': [
            {'name': 'az_0'},
            {'name': 'az_p15', 'azimuth_deg': 15},
            {'name': 'az_m15', 'azimuth_deg': -15},
            {'name': 'az_p30', 'azimuth_deg': 30},
            {'name': 'az_m30', 'azimuth_deg': -30},
        ]
    },
    'azimuth_sweep3': {
        'camera': 'agentview',
        'viewpoints': [
            {'name': 'az_0'},
            {'name': 'az_p30', 'azimuth_deg': 30},
            {'name': 'az_m30', 'azimuth_deg': -30},
        ]
    },
}

# ---------------------------------------------------------------------------
# pose math (numpy rotation matrices; robosuite quats are xyzw, mujoco wxyz)
# ---------------------------------------------------------------------------

def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def _rot_rpy(rpy):
    # extrinsic xyz: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    rx, ry, rz = rpy
    return _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)

def _look_at_quat(pos, target):
    """xyzw quaternion orienting the camera (looking along -z) at target."""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    z_axis = -forward  # mujoco cameras look along -z
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # camera nearly straight down/up: fall back to world x
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rmat = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns = cam axes
    return T.mat2quat(rmat)  # xyzw

def _compute_perturbed_pose(base_pos, base_quat_wxyz, spec):
    """
    Absolute camera pose from the base (dataset) pose. Applied in order:
    azimuth orbit + look-at, local elevation pitch, pos_delta / euler_delta_deg.
    Empty spec = base pose (identity, used to restore the camera).
    Returns (pos, quat_wxyz).
    """
    pos = np.asarray(base_pos, dtype=np.float64).copy()
    base_quat = T.convert_quat(
        np.asarray(base_quat_wxyz, dtype=np.float64), to='xyzw')

    has_move = any(k in spec for k in
        ('azimuth_deg', 'elevation_deg', 'pos_delta', 'euler_delta_deg'))
    if not has_move:
        return pos, np.asarray(base_quat_wxyz, dtype=np.float64).copy()

    azimuth_deg = spec.get('azimuth_deg', 0.0)
    elevation_deg = spec.get('elevation_deg', 0.0)
    if azimuth_deg != 0.0 or elevation_deg != 0.0:
        pos = _rot_z(math.radians(azimuth_deg)) @ pos
        # base camera look-at point (1 m along the camera forward axis, -z)
        forward = T.quat2mat(base_quat) @ np.array([0.0, 0.0, -1.0])
        target = spec.get('look_at', None)
        if target is None:
            target = np.asarray(base_pos, dtype=np.float64) + forward
        quat = _look_at_quat(pos, np.asarray(target, dtype=np.float64))
        if elevation_deg != 0.0:
            pitch = T.mat2quat(_rot_x(math.radians(elevation_deg)))
            # post-multiplied local pitch: R = R_lookat @ R_x
            quat = T.quat_multiply(quat, pitch)
    else:
        quat = base_quat

    if 'pos_delta' in spec:
        pos = pos + np.asarray(spec['pos_delta'], dtype=np.float64)
    if 'euler_delta_deg' in spec:
        delta = T.mat2quat(_rot_rpy(np.radians(
            np.asarray(spec['euler_delta_deg'], dtype=np.float64))))
        quat = T.quat_multiply(delta, quat)

    return pos, T.convert_quat(quat, to='wxyz')


class ViewpointImageWrapper(RobomimicImageWrapper):
    """Script-local wrapper: applies a camera viewpoint at every reset."""

    def __init__(self, *args, serve_obs_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewpoint = None
        self.serve_obs_key = serve_obs_key
        self._base_camera_poses = dict()

    def get_observation(self, raw_obs=None):
        # A policy trained on a multi-view dataset expects a `view_XX_image`
        # key, but the live env only ever produces its camera-derived key
        # (`agentview_image`). Alias it here rather than at the policy call
        # site: the base class builds obs by indexing raw_obs with the
        # shape_meta keys, so the alias has to exist before that loop runs.
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        if self.serve_obs_key is not None and self.serve_obs_key not in raw_obs:
            raw_obs = dict(raw_obs)
            raw_obs[self.serve_obs_key] = raw_obs[self.render_obs_key]
        return super().get_observation(raw_obs)

    def set_viewpoint(self, viewpoint):
        self.viewpoint = viewpoint

    def is_success(self):
        # passthrough so AsyncVectorEnv.call('is_success') works through
        # the gym.Wrapper chain (this class is a bare gym.Env).
        return self.env.is_success()

    def reset(self):
        obs = super().reset()
        if self.viewpoint is not None:
            self._apply_viewpoint()
            # force re-render at the moved pose (avoids a stale first frame)
            raw_obs = self.env.get_observation()
            obs = self.get_observation(raw_obs)
        return obs

    def _apply_viewpoint(self):
        spec = dict(self.viewpoint)
        camera = spec.pop('camera', 'agentview')
        sim = self.env.env.sim
        cid = sim.model.camera_name2id(camera)
        if camera not in self._base_camera_poses:
            # capture the base (dataset) pose on first use, before any move
            self._base_camera_poses[camera] = (
                sim.model.cam_pos[cid].copy(),
                sim.model.cam_quat[cid].copy()
            )
        base_pos, base_quat_wxyz = self._base_camera_poses[camera]
        pos, quat_wxyz = _compute_perturbed_pose(base_pos, base_quat_wxyz, spec)
        sim.model.cam_pos[cid] = pos
        sim.model.cam_quat[cid] = quat_wxyz
        sim.forward()


# ---------------------------------------------------------------------------
# eval harness
# ---------------------------------------------------------------------------

def undo_transform_action(rotation_transformer, action):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3:3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([
        pos, rot, gripper
    ], axis=-1)

    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction


def run_novel_view_eval(policy,
        output_dir,
        shape_meta,
        dataset_path,
        preset,
        n_test=50,
        n_test_vis=4,
        n_envs=28,
        n_obs_steps=2,
        n_action_steps=8,
        max_steps=400,
        render_obs_key='agentview_image',
        serve_obs_key=None,
        past_action=False,
        abs_action=False,
        fps=10,
        crf=22,
        test_start_seed=100000,
        tqdm_interval_sec=1.0):
    """
    Roll a policy out at every (viewpoint, seed) pair and return the log dict
    (with wandb.Video values for recorded rollouts). Same seeds across
    viewpoints = paired evaluation.
    """
    device = policy.device
    dataset_path = os.path.expanduser(dataset_path)
    n_envs = max(int(n_envs), 1)

    # read env metadata from dataset (replicates RobomimicImageRunner)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    # disable object state observation
    env_meta['env_kwargs']['use_object_obs'] = False

    rotation_transformer = None
    if abs_action:
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    robosuite_fps = 20
    steps_per_render = max(robosuite_fps // fps, 1)

    def env_fn():
        robomimic_env = create_env(
            env_meta=env_meta,
            shape_meta=shape_meta
        )
        # Robosuite's hard reset causes excessive memory consumption.
        # Disabled to run more envs.
        robomimic_env.env.hard_reset = False
        return MultiStepWrapper(
            VideoRecordingWrapper(
                ViewpointImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key,
                    serve_obs_key=serve_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=crf,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=steps_per_render
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )

    # For each process the OpenGL context can only be initialized once.
    # Since AsyncVectorEnv uses fork to create worker processes, a separate
    # env_fn that does not create an OpenGL context is needed for spaces.
    def dummy_env_fn():
        robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                enable_render=False
            )
        return MultiStepWrapper(
            VideoRecordingWrapper(
                ViewpointImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key,
                    serve_obs_key=serve_obs_key
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=crf,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=steps_per_render
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )

    env_fns = [env_fn] * n_envs
    env_seeds = list()
    env_prefixs = list()
    env_init_fn_dills = list()

    # one env slot per (viewpoint, seed); same seeds across viewpoints
    # (paired evaluation)
    for vp in preset['viewpoints']:
        vp = dict(vp)
        vp['camera'] = preset['camera']
        vp_name = vp['name']
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed,
                    enable_render=enable_render, vp=vp, vp_name=vp_name):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', vp_name, wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset + set viewpoint
                assert isinstance(env.env.env, ViewpointImageWrapper)
                env.env.env.init_state = None
                env.env.env.set_viewpoint(vp)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append(f'test/{vp_name}/')
            env_init_fn_dills.append(dill.dumps(init_fn))

    env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

    # plan for rollout
    n_inits = len(env_init_fn_dills)
    n_chunks = math.ceil(n_inits / n_envs)

    # allocate data
    all_video_paths = [None] * n_inits
    all_rewards = [None] * n_inits
    all_successes = [None] * n_inits

    for chunk_idx in range(n_chunks):
        start = chunk_idx * n_envs
        end = min(n_inits, start + n_envs)
        this_global_slice = slice(start, end)
        this_n_active_envs = end - start
        this_local_slice = slice(0, this_n_active_envs)

        this_init_fns = env_init_fn_dills[this_global_slice]
        n_diff = n_envs - len(this_init_fns)
        if n_diff > 0:
            this_init_fns.extend([env_init_fn_dills[0]] * n_diff)
        assert len(this_init_fns) == n_envs

        # init envs
        env.call_each('run_dill_function',
            args_list=[(x,) for x in this_init_fns])

        # start rollout
        obs = env.reset()
        last_action = None
        policy.reset()

        env_name = env_meta['env_name']
        pbar = tqdm.tqdm(total=max_steps,
            desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}",
            leave=False, mininterval=tqdm_interval_sec)

        done = False
        while not done:
            # create obs dict
            np_obs_dict = dict(obs)
            if past_action and (last_action is not None):
                np_obs_dict['past_action'] = last_action[
                    :, -(n_obs_steps - 1):].astype(np.float32)

            # device transfer
            obs_dict = dict_apply(np_obs_dict,
                lambda x: torch.from_numpy(x).to(
                    device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action']
            if not np.all(np.isfinite(action)):
                print(action)
                raise RuntimeError("Nan or Inf action")

            # step env
            env_action = action
            if abs_action:
                env_action = undo_transform_action(rotation_transformer, action)

            obs, reward, done, info = env.step(env_action)
            done = np.all(done)
            last_action = action

            # update pbar
            pbar.update(action.shape[1])
        pbar.close()

        # collect data for this round
        all_video_paths[this_global_slice] = env.render()[this_local_slice]
        all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        all_successes[this_global_slice] = env.call('is_success')[this_local_slice]
    # clear out video buffer
    _ = env.reset()

    # log (same structure as RobomimicImageRunner.run, plus success_rate)
    max_rewards = collections.defaultdict(list)
    successes = collections.defaultdict(list)
    log_data = dict()
    for i in range(n_inits):
        seed = env_seeds[i]
        prefix = env_prefixs[i]
        max_reward = np.max(all_rewards[i])
        max_rewards[prefix].append(max_reward)
        log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

        succ = all_successes[i]
        if succ is not None:
            success = float(succ['task'])
            successes[prefix].append(success)
            log_data[prefix + f'success_{seed}'] = success

        # visualize sim
        video_path = all_video_paths[i]
        if video_path is not None:
            sim_video = wandb.Video(video_path)
            log_data[prefix + f'sim_video_{seed}'] = sim_video

    # log aggregate metrics
    for prefix, value in max_rewards.items():
        log_data[prefix + 'mean_score'] = float(np.mean(value))
    for prefix, value in successes.items():
        log_data[prefix + 'success_rate'] = float(np.mean(value))

    return log_data


def to_json_log(log_data):
    """Convert wandb.Video values to paths so the log is json-serializable."""
    json_log = dict()
    for key, value in log_data.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    return json_log


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--preset', 'preset_name',
    type=click.Choice(list(PRESETS.keys())), default='azimuth_sweep5')
@click.option('--n-test', type=int, default=None,
    help='test rollouts per viewpoint (default: checkpoint config value)')
@click.option('--n-test-vis', type=int, default=None,
    help='videos per viewpoint (default: checkpoint config value)')
@click.option('--n-envs', type=int, default=None,
    help='parallel envs (default: checkpoint config value)')
@click.option('--serve-obs-key', default=None,
    help='also serve the rendered camera image under this obs key, for '
         'policies trained on a multi-view dataset (e.g. view_06_image)')
def main(checkpoint, output_dir, device, preset_name, n_test, n_test_vis, n_envs,
        serve_obs_key):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint (same as eval.py)
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # env runner settings from the checkpoint config (resolved at save time)
    er = cfg.task.env_runner

    # The live env can only produce image keys robosuite actually renders (the
    # camera-derived name, e.g. agentview_image), and robomimic additionally
    # drops any rgb key missing from the obs-modality mapping built from
    # shape_meta. A policy trained on a multi-view dataset names its view
    # itself (e.g. view_06_image), so the env's shape_meta must carry the
    # rendered key as well; --serve-obs-key aliases one to the other in
    # ViewpointImageWrapper.get_observation.
    env_shape_meta = er.shape_meta
    if serve_obs_key is not None and er.render_obs_key not in env_shape_meta['obs']:
        env_shape_meta = copy.deepcopy(env_shape_meta)
        env_shape_meta['obs'][er.render_obs_key] = dict(
            shape=env_shape_meta['obs'][serve_obs_key]['shape'], type='rgb')

    log_data = run_novel_view_eval(
        policy=policy,
        output_dir=output_dir,
        shape_meta=env_shape_meta,
        dataset_path=er.dataset_path,
        preset=PRESETS[preset_name],
        n_test=er.n_test if n_test is None else n_test,
        n_test_vis=er.n_test_vis if n_test_vis is None else n_test_vis,
        n_envs=er.n_envs if n_envs is None else n_envs,
        n_obs_steps=er.n_obs_steps,
        n_action_steps=er.n_action_steps,
        max_steps=er.max_steps,
        render_obs_key=er.render_obs_key,
        serve_obs_key=serve_obs_key,
        past_action=er.past_action,
        abs_action=er.abs_action,
        fps=er.fps,
        crf=er.crf,
        test_start_seed=er.test_start_seed)

    # dump log to json (same format as eval.py)
    json_log = to_json_log(log_data)
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    # print summary
    summary = {
        prefix: {
            'mean_score': json_log[prefix + 'mean_score'],
            'success_rate': json_log.get(prefix + 'success_rate', None)
        }
        for prefix in sorted(set(
            k.rsplit('/', 1)[0] + '/' for k in json_log if k.endswith('mean_score')))
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
