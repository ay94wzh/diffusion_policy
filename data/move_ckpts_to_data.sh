#!/bin/bash
# Phase 0 of the recovery plan: relocate every checkpoint under data/outputs/ to
# the /data volume and leave a symlink behind.
#
# Why a MOVE and not a delete: NOTES.md forbids deleting a cell's weights before
# its latent has been probed, and forbids deleting m3off outright. / is at 100%
# (9.7 GB free) while /data has 9.4 TB.
#
# Why rsync --remove-source-files and not mv: this is a cross-device transfer of
# 103.6 GB. rsync verifies each file as it lands and only then unlinks the
# source, so an interrupted run leaves both sides readable and the script
# re-runnable. The per-dir byte count is compared before the source is removed.
#
# Run dirs keep their logs.json.txt / media/ / .hydra/ in place -- those are what
# the git whitelists track.
set -u
cd /home/zihan/repos/diffusion_policy || exit 1
STORE=/data/zihan/ckpt_store
LOG=data/move_ckpts.log
mkdir -p "$STORE"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

dir_bytes() { find "$1" -name '*.ckpt' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}'; }

moved=0; skipped=0; failed=0
for src in data/outputs/run_*/checkpoints; do
    run=$(basename "$(dirname "$src")")

    # already a symlink -> a previous pass did this dir
    if [ -L "$src" ]; then
        skipped=$((skipped+1)); continue
    fi
    [ -d "$src" ] || continue

    before=$(dir_bytes "$src")
    if [ "$before" -eq 0 ]; then
        say "SKIP $run (no .ckpt files)"; skipped=$((skipped+1)); continue
    fi

    dest="$STORE/$run/checkpoints"
    mkdir -p "$dest"
    if ! rsync -a --remove-source-files "$src"/ "$dest"/ 2>>"$LOG"; then
        say "FAIL $run (rsync error -- source left intact)"; failed=$((failed+1)); continue
    fi

    after=$(dir_bytes "$dest")
    if [ "$after" -ne "$before" ]; then
        say "FAIL $run (bytes $after != $before -- source left intact, NOT removing)"; failed=$((failed+1)); continue
    fi

    # only now is the source safe to drop: it holds no .ckpt any more, just the
    # emptied directory tree
    find "$src" -type f -o -type d | sort -r | xargs -r rmdir 2>/dev/null
    rmdir "$src" 2>/dev/null
    ln -s "$dest" "$src"
    say "OK   $run  $(numfmt --to=iec "$before") -> $dest"
    moved=$((moved+1))
done

say "DONE moved=$moved skipped=$skipped failed=$failed"
df -h / /data | tail -3 | tee -a "$LOG"
