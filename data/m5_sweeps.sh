#!/bin/bash
# Phase 1 of the recovery plan: sweep and screen the three runs that
# data/m5_campaign.sh trained to completion on 2026-09-26 but never evaluated.
#
#   (a) lift   + m3on            /data/zihan/runs/run_lift_m3on_s42_200ep      slots 7
#   (b) square + [1,5], seed 43  /data/zihan/runs/run_square_m3v15_s43_200ep   slots 7
#   (c) square + pm60 view pool  /data/zihan/runs/run_square_m3pm60_s42_200ep  slots 5
#
# THE BUG THIS SCRIPT EXISTS TO NOT REPEAT. m5_campaign.sh trained all three,
# then its ckpt_loads guard reported CKPT_LOAD_FAILED for all three and skipped
# every sweep. The checkpoints were never corrupt: the guard called a bare
# `python`, which on this box is conda BASE and has no torch, and it swallowed
# the error into `grep -q`. Three 4.6 GB loads "failing" inside one second is
# what gives it away -- a real load takes ~30 s. Hence, below:
#   * PY is pinned to the robodiff interpreter (the convention every other
#     driver script in data/ already follows)
#   * the guard PRINTS the error it saw instead of hiding it
#   * a guard failure is reported as UNVERIFIED, not as a corrupt checkpoint
#
# Idempotent: every stage is skipped if its output already exists, because
# eval_novel_view.py blocks on a click.confirm prompt when -o exists.
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
PY=/home/zihan/anaconda3/envs/robodiff/bin/python
LOG=data/m5_sweeps.log
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# Load a checkpoint and say what happened.
# No pipe on stdout: the exit status must be python's own, or the guard cannot
# fail (a `... | tail` pipeline reports tail's status, which is always 0).
# A traceback goes to $LOG rather than into the captured text.
ckpt_check() {  # $1 = checkpoint path
    "$PY" - "$1" 2>>"$LOG" <<'PYEOF'
import sys, torch, dill
p = torch.load(sys.argv[1], pickle_module=dill, map_location='cpu')
print(f'LOAD_OK state_dicts={len(p["state_dicts"])} '
      f'task={p["cfg"]["task"]["task_name"]} '
      f'pool={p["cfg"]["task"]["dataset"].get("view_pool")} '
      f'range={p["cfg"]["task"]["dataset"].get("view_count_range")}')
PYEOF
}

# One run: two sweeps, the summary, then the collapse screens.
# $1 run-dir name  $2 task  $3 slots  $4 gpu  $5 screen-range  $6 cell
finish() {
    local name=$1 task=$2 slots=$3 gpu=$4 scrange=$5 cell=$6
    # split across two `local`s on purpose: under `set -u` bash expands every
    # word of a `local` before performing any of its assignments, so
    # `local d=... C=$d/...` dies on an unbound `d`.
    local d=/data/zihan/runs/run_$name
    local C=$d/checkpoints/latest.ckpt
    local interp=data/eval_interp_${task}_${cell} el=data/eval_el_${task}_${cell}

    say "${task}_${cell}: checking $C"
    if ! out=$(ckpt_check "$C"); then
        say "${task}_${cell}_CHECK_UNVERIFIED (the interpreter, not the data -- see header)"
        say "  raw: $out"; return 1
    fi
    say "${task}_${cell}_CKPT $out"

    # ---- azimuth_interp: 11 viewpoints, 50 paired episodes each (~46 min) ----
    if [ -f "$interp/eval_log.json" ]; then
        say "${task}_${cell}_INTERP_SKIPPED (already present)"
    else
        say "${task}_${cell}_INTERP_START"
        "$PY" eval_novel_view.py -c "$C" -o "$interp" -d "$gpu" \
            --preset azimuth_interp --m3-slots "$slots" --eef-hist-steps 4 \
            --n-envs 14 --n-test-vis 0 < /dev/null \
            >> data/sweep_${task}_${cell}.log 2>&1
        say "${task}_${cell}_INTERP_DONE exit=$?"
    fi

    # ---- elevation_az0: the off-manifold orbit (~12 min) ----
    if [ -f "$el/eval_log.json" ]; then
        say "${task}_${cell}_ELEVATION_SKIPPED (already present)"
    else
        say "${task}_${cell}_ELEVATION_START"
        "$PY" eval_novel_view.py -c "$C" -o "$el" -d "$gpu" \
            --preset elevation_az0 --m3-slots "$slots" --eef-hist-steps 4 \
            --n-envs 14 --n-test-vis 0 < /dev/null \
            >> data/sweep_${task}_${cell}.log 2>&1
        say "${task}_${cell}_ELEVATION_DONE exit=$?"
    fi

    "$PY" summarize_novel_view.py "$interp" "$el" \
        > data/eval_${task}_${cell}_summary.txt 2>&1
    say "${task}_${cell}_SUMMARY_DONE -> data/eval_${task}_${cell}_summary.txt"

    # ---- collapse screen, matched range + its own random-init baseline ----
    # The baseline is per-architecture and must be measured, never borrowed;
    # a matched --view-count-range is required or cells are not comparable.
    if [ -f "data/screen_collapse/${task}_${cell}_latest.json" ]; then
        say "${task}_${cell}_SCREEN_SKIPPED (already present)"
    else
        "$PY" screen_collapse.py -c "$C" -d "$gpu" --view-count-range "$scrange" \
            -o data/screen_collapse/${task}_${cell}_latest.json \
            >> data/screen_${task}_${cell}.log 2>&1
        "$PY" screen_collapse.py -c "$C" -d "$gpu" --view-count-range "$scrange" --random-init \
            -o data/screen_collapse/${task}_${cell}_RANDOM_INIT.json \
            >> data/screen_${task}_${cell}.log 2>&1
        say "${task}_${cell}_SCREENS_DONE (range $scrange)"
    fi
}

# ---- stage 1: (a) and (b) concurrently, one per GPU ----
say "SWEEPS_START (a) lift_m3on on cuda:0, (b) square_m3v15_s43 on cuda:1"
finish lift_m3on_s42_200ep     lift   7 cuda:0 7,7 m3on       & P1=$!
finish square_m3v15_s43_200ep  square 7 cuda:1 7,7 m3v15_s43  & P2=$!
wait $P1; say "A_EXIT=$?"
wait $P2; say "B_EXIT=$?"

# ---- stage 2: (c), whose pool is five views so its screen range is 5,5 ----
finish square_m3pm60_s42_200ep square 5 cuda:0 5,5 m3pm60
say "C_EXIT=$?"

say "SWEEPS_DONE"
