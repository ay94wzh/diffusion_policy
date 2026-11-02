"""Read-only audit of eval diffusion-noise pairing (no evaluator changes).

The plan doc (ENCODER_B1_RESIDUAL_TOKEN_PLAN.md section 6) flags that
`eval_novel_view.py` pairs environment seeds but does not visibly pair diffusion
noise by episode/control step. This script measures the current behaviour
without changing the evaluator:

  - wraps `policy.reset` to mark a chunk and `policy.conditional_sample` to log
    the CPU/CUDA RNG digests immediately before each sampling call;
  - runs one viewpoint at a time through `eval_novel_view.run_novel_view_eval`
    with a small test set, for production-like batching (`n_envs=4`, two
    chunks) and serialized control (`n_envs=1`);
  - compares records keyed by `(chunk, call_in_chunk)` across scenarios.

Scenarios:
  same   -- the same viewpoint twice with a fresh seed (reproducibility);
  az_el  -- `az_0` vs `el_0`, both no-rotation presets (the doc's 0.80 vs 0.90
            consistency check);
  fresh  -- `az_0` vs `az_p30`, each with a fresh seed (does a fresh per-
            viewpoint seed pair the noise?);
  shared -- `az_0` then `az_p30` on the production shared stream (unpaired?);
  reseed -- optional demonstration: seed the global RNG per sampling call; the
            streams match until the batch structure diverges.

This script needs miroc-server (robosuite + a checkpoint + the hdf5). It does
not modify `eval_novel_view.py` or any eval protocol. It writes a JSON report.

Usage:
    python audit_eval_noise.py -c <checkpoint> -d cuda:0 \
        --out data/audit_eval_noise_b1.json
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
import tempfile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import eval_novel_view as env_mod


def _digest(data):
    return hashlib.sha256(data).hexdigest()[:16]


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_policy(checkpoint, device, output_dir):
    payload = torch.load(
        open(checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.to(torch.device(device))
    policy.eval()
    return cfg, policy


def _env_shape_meta(cfg, m3_slots):
    er = cfg.task.env_runner
    env_shape_meta = copy.deepcopy(er.shape_meta)
    if m3_slots > 0:
        OmegaConf.set_struct(env_shape_meta['obs'], False)
        slot_keys = [k for k, a in env_shape_meta['obs'].items()
                     if a.get('type', 'low_dim') == 'rgb']
        assert len(slot_keys) == m3_slots, (len(slot_keys), m3_slots)
        render_shape = env_shape_meta['obs'][slot_keys[0]]['shape']
        for k in slot_keys:
            del env_shape_meta['obs'][k]
        env_shape_meta['obs'][er.render_obs_key] = dict(
            shape=render_shape, type='rgb')
    return env_shape_meta, er


def _viewpoint_spec(name):
    for preset in env_mod.PRESETS.values():
        for vp in preset['viewpoints']:
            if vp['name'] == name:
                return preset['camera'], vp
    raise KeyError(f'viewpoint {name!r} not found in any preset')


def _install_wrappers(policy, records, state):
    """Record RNG state before each conditional_sample call.

    `policy.reset` marks a chunk (it is called once per rollout chunk by
    run_novel_view_eval); `conditional_sample` marks the control-step call.
    """
    original_reset = policy.reset
    original_sample = policy.conditional_sample

    def reset_wrapper(*args, **kwargs):
        state['chunk'] += 1
        state['call'] = 0
        return original_reset(*args, **kwargs)

    def sample_wrapper(condition_data, condition_mask, *args, **kwargs):
        if state.get('reseed_each_call', False):
            torch.manual_seed(
                int(state['base_seed'])
                + 100000 * int(state['chunk'])
                + int(state['call']))
        rec = dict(
            chunk=int(state['chunk']),
            call=int(state['call']),
            batch=int(condition_data.shape[0]),
            shape=tuple(int(x) for x in condition_data.shape),
            cpu=_digest(torch.get_rng_state().numpy().tobytes()),
            cuda=None,
        )
        if condition_data.is_cuda:
            rec['cuda'] = _digest(
                torch.cuda.get_rng_state(condition_data.device)
                .cpu().numpy().tobytes())
        records.append(rec)
        state['call'] += 1
        return original_sample(
            condition_data, condition_mask, *args, **kwargs)

    policy.reset = reset_wrapper
    policy.conditional_sample = sample_wrapper

    def restore():
        policy.reset = original_reset
        policy.conditional_sample = original_sample

    return restore


def _run_one(policy, er, env_shape_meta, vp_name, n_test, n_envs,
             output_dir, records, state):
    camera, vp = _viewpoint_spec(vp_name)
    preset = {'camera': camera, 'viewpoints': [vp]}
    # Each run_one is its own rollout; chunk/call keys start at (0, 0) so the
    # two arms of a scenario are directly comparable.
    state['chunk'] = -1
    state['call'] = 0
    records.clear()
    log = env_mod.run_novel_view_eval(
        policy=policy,
        output_dir=output_dir,
        shape_meta=env_shape_meta,
        dataset_path=er.dataset_path,
        preset=preset,
        n_test=n_test,
        n_test_vis=0,
        n_envs=n_envs,
        n_obs_steps=er.n_obs_steps,
        n_action_steps=er.n_action_steps,
        max_steps=er.max_steps,
        render_obs_key=er.render_obs_key,
        serve_obs_key=None,
        m3_slots=11,
        eef_hist_steps=4,
        serve_eef_hist_base=False,
        past_action=er.past_action,
        abs_action=er.abs_action,
        fps=er.fps,
        crf=er.crf,
        test_start_seed=er.test_start_seed,
    )
    return log, [dict(r) for r in records]


def _compare_records(a, b):
    map_a = {(r['chunk'], r['call']): r for r in a}
    map_b = {(r['chunk'], r['call']): r for r in b}
    common = sorted(set(map_a) & set(map_b))
    mismatches = []
    for key in common:
        ra, rb = map_a[key], map_b[key]
        if ra['cpu'] != rb['cpu'] or ra['cuda'] != rb['cuda']:
            mismatches.append(dict(key=key, a=ra, b=rb))
    return dict(
        n_a=len(a), n_b=len(b), n_common=len(common),
        n_match=len(common) - len(mismatches),
        first_mismatch=mismatches[0] if mismatches else None,
        mismatches=mismatches[:5],
    )


def _success(log, vp):
    return log.get(f'test/{vp}/success_rate')


def _per_seed(log, vp):
    prefix = f'test/{vp}/success_'
    rate = f'test/{vp}/success_rate'
    out = {}
    for key, value in log.items():
        if key.startswith(prefix) and key != rate:
            out[key[len(prefix):]] = float(value)
    return out


def _run_scenario(name, policy, er, env_shape_meta, viewpoints,
                  n_test, n_envs, seed, reset_between, output_dir,
                  records, state, reseed_each_call=False):
    state['reseed_each_call'] = bool(reseed_each_call)
    state['base_seed'] = int(seed)
    results = []
    for vp in viewpoints:
        if reset_between:
            _set_seed(seed)
        log, recs = _run_one(
            policy, er, env_shape_meta, vp, n_test, n_envs,
            output_dir, records, state)
        results.append(dict(viewpoint=vp, log=log, records=recs))
    state['reseed_each_call'] = False

    out = {'viewpoints': [r['viewpoint'] for r in results]}
    for i, result in enumerate(results):
        vp = result['viewpoint']
        out[f'success_rate_{vp}_{i}'] = _success(result['log'], vp)
        out[f'per_seed_{vp}_{i}'] = _per_seed(result['log'], vp)
    if len(results) == 2:
        out['noise'] = _compare_records(results[0]['records'],
                                        results[1]['records'])
        if results[0]['viewpoint'] == results[1]['viewpoint']:
            out['same_success'] = (
                _success(results[0]['log'], results[0]['viewpoint']) ==
                _success(results[1]['log'], results[1]['viewpoint']))
        if {results[0]['viewpoint'], results[1]['viewpoint']} == {'az_0', 'el_0'}:
            out['consistency_success_equal'] = (
                _success(results[0]['log'], results[0]['viewpoint']) ==
                _success(results[1]['log'], results[1]['viewpoint']))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True)
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('--out', default='data/audit_eval_noise.json')
    parser.add_argument('--n-test', type=int, default=8)
    parser.add_argument('--n-envs', type=int, default=4,
                        help='production-like batch (two chunks at n_test=8)')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--no-reseed', action='store_true',
                        help='skip the optional per-call reseed demonstration')
    args = parser.parse_args()

    output_dir = tempfile.mkdtemp(prefix='audit_eval_noise_')
    cfg, policy = _load_policy(args.checkpoint, args.device, output_dir)
    env_shape_meta, er = _env_shape_meta(cfg, m3_slots=11)

    records = []
    state = dict(chunk=-1, call=0, reseed_each_call=False,
                 base_seed=args.seed)
    restore = _install_wrappers(policy, records, state)
    try:
        report = dict(
            checkpoint=args.checkpoint,
            n_test=args.n_test,
            seed=args.seed,
            batching={},
        )
        for n_envs in sorted({int(args.n_envs), 1}):
            key = f'n_envs={n_envs}'
            entry = dict(
                same=_run_scenario(
                    'same', policy, er, env_shape_meta, ['az_0', 'az_0'],
                    args.n_test, n_envs, args.seed, True, output_dir,
                    records, state),
                az_el=_run_scenario(
                    'az_el', policy, er, env_shape_meta, ['az_0', 'el_0'],
                    args.n_test, n_envs, args.seed, True, output_dir,
                    records, state),
                fresh=_run_scenario(
                    'fresh', policy, er, env_shape_meta, ['az_0', 'az_p30'],
                    args.n_test, n_envs, args.seed, True, output_dir,
                    records, state),
                shared=_run_scenario(
                    'shared', policy, er, env_shape_meta, ['az_0', 'az_p30'],
                    args.n_test, n_envs, args.seed, False, output_dir,
                    records, state),
            )
            if not args.no_reseed:
                entry['reseed'] = _run_scenario(
                    'reseed', policy, er, env_shape_meta,
                    ['az_0', 'az_p30'], args.n_test, n_envs, args.seed,
                    True, output_dir, records, state,
                    reseed_each_call=True)
            report['batching'][key] = entry
    finally:
        restore()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.',
                exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)

    # Concise stdout summary.
    for key, entry in report['batching'].items():
        print(f'[{key}]')
        for name in ('same', 'az_el', 'fresh', 'shared', 'reseed'):
            if name not in entry:
                continue
            noise = entry[name].get('noise', {})
            print(f'  {name}: calls={noise.get("n_a")}/{noise.get("n_b")} '
                  f'match={noise.get("n_match")}/{noise.get("n_common")} '
                  f'first_mismatch={noise.get("first_mismatch", {}).get("key") if noise.get("first_mismatch") else None} '
                  f'success_equal={entry[name].get("same_success", entry[name].get("consistency_success_equal"))}')
        azel = entry['az_el']
        print('  az_0 vs el_0 success: '
              f'{azel.get("success_rate_az_0_0")} vs '
              f'{azel.get("success_rate_el_0_1")} '
              f'(equal={azel.get("consistency_success_equal")})')
    print(f'\nreport -> {args.out}')
    print('Interpretation: same=True means repeated fresh-seed eval is '
          'reproducible; az_el equal=True means the no-rotation presets are '
          'consistent; fresh match==common means a fresh seed pairs the noise; '
          'shared first_mismatch=0 means the production stream is unpaired '
          'from the first call.')


if __name__ == '__main__':
    main()
