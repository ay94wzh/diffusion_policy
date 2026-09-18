"""Rollout runner for M3's view-conditioned encoder (one live camera).

`RobomimicImageRunner` cannot serve M3's obs: the policy consumes K view-slot
keys, cam vectors, an activity mask and an EE-pose history, none of which the
live env produces. This module adds them.

Why a subclass that copies `__init__`
-------------------------------------
Three constraints force it:

1. `RobomimicImageRunner.__init__` builds its `env_fn` / `dummy_env_fn`
   closures *internally* (robomimic_image_runner.py:89-154) and instantiates
   `RobomimicImageWrapper` by name, so there is no hook to swap the wrapper.
2. The runner's `init_fn`s assert `isinstance(env.env.env, RobomimicImageWrapper)`
   (lines 183, 210), so the wrapper must **subclass** `RobomimicImageWrapper`,
   not wrap it.
3. The obs-key set must match the keys the dataset fitted normalizers for, or
   `predict_action`'s `LinearNormalizer` raises `KeyError` -- and the env-side
   `shape_meta` must contain exactly the rgb keys robosuite can actually render,
   because `create_env` builds robomimic's modality mapping from it.

So `__init__` below is a copy of the parent's body with those three changes;
everything else (dill'd init_fns, AsyncVectorEnv, attributes) is verbatim.

The live env has ONE camera, so every rollout is an **N=1** inference at the
training pose -- which is M5's setting, and the reason N is randomized during
training so that N=1 is in distribution. Slot 0 carries the live frame and the
live pose; the other slots are zeros and masked off. For real (novel-view)
numbers use `eval_novel_view.py --m3-slots K`, not the in-training rollout.
"""
import collections
import pathlib

import copy
import dill
import numpy as np
from gym import spaces

from diffusion_policy.env_runner.robomimic_image_runner import (
    RobomimicImageRunner, create_env)
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from diffusion_policy.dataset.multiview_image_dataset import (
    EEF_HIST_STEP_DIM, eef_hist_to_cam)
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper, VideoRecorder)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import robomimic.utils.file_utils as FileUtils


class CamKeyImageWrapper(RobomimicImageWrapper):
    """Serves M3's slot / camera / mask / EE-history keys from one live camera."""

    def __init__(self, *args, m3_slots=0, eef_hist_steps=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.m3_slots = int(m3_slots)
        self.eef_hist_steps = int(eef_hist_steps)
        self._eef_hist = collections.deque(maxlen=max(self.eef_hist_steps, 1))
        if self.m3_slots > 0:
            # Registering on the observation space is required (not merely
            # tidy): MultiStepWrapper and gym's shared-memory writer both
            # whitelist by space key, so unregistered keys never reach the
            # policy. Bounds are not enforced on write -- only shape and dtype.
            render_shape = tuple(self.observation_space[self.render_obs_key].shape)
            for k in range(self.m3_slots):
                self.observation_space[f'view_slot_{k:02d}_image'] = spaces.Box(
                    low=0.0, high=1.0, shape=render_shape, dtype=np.float32)
                self.observation_space[f'view_slot_{k:02d}_cam'] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
            self.observation_space['view_mask'] = spaces.Box(
                low=0.0, high=1.0, shape=(self.m3_slots,), dtype=np.float32)
            self.observation_space['view_eef_hist'] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.m3_slots, EEF_HIST_STEP_DIM * self.eef_hist_steps),
                dtype=np.float32)

    def _cam_name(self):
        # the render key is `<camera>_image`
        return self.render_obs_key[:-len('_image')]

    def _serve_m3(self, raw_obs):
        frame = raw_obs[self.render_obs_key]
        h, w = frame.shape[-2:]
        sim = self.env.env.sim
        cid = sim.model.camera_name2id(self._cam_name())
        # read back FROM THE SIM: the published pose is then exactly the pose
        # that rendered the frame (mujoco cam_quat is already wxyz)
        cam = np.concatenate([
            np.asarray(sim.model.cam_pos[cid], dtype=np.float64),
            np.asarray(sim.model.cam_quat[cid], dtype=np.float64),
            [float(sim.model.cam_fovy[cid]), float(h), float(w)],
        ]).astype(np.float32)

        pose = (np.asarray(raw_obs['robot0_eef_pos'], dtype=np.float64),
                np.asarray(raw_obs['robot0_eef_quat'], dtype=np.float64),
                np.asarray(raw_obs['robot0_gripper_qpos'], dtype=np.float64))
        if len(self._eef_hist) == 0:
            # seed by repeating the first pose: the same pad rule
            # MultiViewImageDataset._eef_history applies at the window start
            for _ in range(self.eef_hist_steps):
                self._eef_hist.append(pose)
        else:
            self._eef_hist.append(pose)
        pos = np.stack([p for p, _, _ in self._eef_hist])
        quat = np.stack([q for _, q, _ in self._eef_hist])
        grip = np.stack([g for _, _, g in self._eef_hist])
        hist = eef_hist_to_cam(pos, quat, grip, cam[:3], cam[3:7]).reshape(-1)

        mask = np.zeros(self.m3_slots, dtype=np.float32)
        mask[0] = 1.0
        for k in range(self.m3_slots):
            live = (k == 0)
            raw_obs[f'view_slot_{k:02d}_image'] = (
                frame if live else np.zeros_like(frame))
            raw_obs[f'view_slot_{k:02d}_cam'] = (
                cam if live else np.zeros(10, dtype=np.float32))
        raw_obs['view_mask'] = mask
        raw_obs['view_eef_hist'] = np.stack([
            hist if k == 0 else np.zeros_like(hist)
            for k in range(self.m3_slots)])

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        raw_obs = dict(raw_obs)
        if self.m3_slots > 0:
            self._serve_m3(raw_obs)
        return super().get_observation(raw_obs)

    def reset(self):
        self._eef_hist.clear()      # reseed per episode
        return super().reset()


class CamKeyImageRunner(RobomimicImageRunner):
    """`RobomimicImageRunner` with `CamKeyImageWrapper` and an env-side
    `shape_meta` carrying only the rgb key robosuite actually renders."""

    def __init__(self,
            output_dir,
            dataset_path,
            shape_meta: dict,
            m3_slots: int = 7,
            eef_hist_steps: int = 4,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        # everything below mirrors RobomimicImageRunner.__init__
        # (robomimic_image_runner.py:46-236), with the three changes documented
        # in this module's docstring. We deliberately do NOT call
        # RobomimicImageRunner.__init__ (it would build the stock wrapper), so
        # the grandparent's state has to be set here.
        super(RobomimicImageRunner, self).__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        dataset_path = str(pathlib.Path(dataset_path).expanduser())
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # CHANGE 1: the env side gets ONE rgb key (the one robosuite renders).
        # The policy's shape_meta keeps the K slots; they are not the env's
        # business -- create_env would otherwise build a modality mapping for
        # cameras that do not exist and EnvRobosuite.get_observation would raise.
        env_shape_meta = copy.deepcopy(shape_meta)
        rgb_keys = [k for k, a in env_shape_meta['obs'].items()
                    if a.get('type', 'low_dim') == 'rgb']
        if len(rgb_keys) > 1 or (rgb_keys and rgb_keys[0] != render_obs_key):
            render_shape = env_shape_meta['obs'][rgb_keys[0]]['shape'] \
                if rgb_keys else [3, 84, 84]
            for k in rgb_keys:
                del env_shape_meta['obs'][k]
            env_shape_meta['obs'][render_obs_key] = dict(
                shape=render_shape, type='rgb')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=env_shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            robomimic_env.env.hard_reset = False
            # CHANGE 2: CamKeyImageWrapper (a RobomimicImageWrapper subclass, so
            # init_fn's isinstance asserts still hold), given the POLICY
            # shape_meta -- it registers the M3 keys itself.
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    CamKeyImageWrapper(
                        env=robomimic_env,
                        shape_meta=env_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        m3_slots=m3_slots,
                        eef_hist_steps=eef_hist_steps
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

        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta,
                    shape_meta=env_shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    CamKeyImageWrapper(
                        env=robomimic_env,
                        shape_meta=env_shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                        m3_slots=m3_slots,
                        eef_hist_steps=eef_hist_steps
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

        # train
        import h5py
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state,
                    enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        import wandb.sdk.data_types.video as wv
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        env.env.file_path = str(filename)
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed,
                enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    import wandb.sdk.data_types.video as wv
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    env.env.file_path = str(filename)
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        # exactly the attributes RobomimicImageRunner.__init__ sets, so the
        # inherited run() works unchanged
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

        self.m3_slots = m3_slots
        self.eef_hist_steps = eef_hist_steps
        self.render_obs_key = render_obs_key
