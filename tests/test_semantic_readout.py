"""CPU-only tests for `analyze_semantic_head.py` (the z-round's semantic
readout): the pure statistics, the collapse predicate, and the npz picker.

Run from the repo root:
    python tests/test_semantic_readout.py

The readout's job is to separate "the trained head learned the invariance"
from "the head mapped everything to one point" (data/z_run.sh risk R1), so
the numbers it prints must be hand-checkable on fixtures where the two cases
are known exactly.
"""
import os
import sys
import tempfile

# repo convention: tests run from the repo root
this_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(this_dir)
os.chdir(repo_root)
sys.path.insert(0, repo_root)

import numpy as np
import torch

from analyze_semantic_head import pairwise_stats, collapse_signature, latest_npz


def test_pairwise_stats():
    # S=2, K=3, D=2. Row 0: 0deg / 90deg / -90deg unit vectors; row 1: all
    # along +x. cosines within a row are then exact by hand.
    import math
    s = torch.tensor([
        [[1., 0.], [0., 1.], [0., -1.]],
        [[1., 0.], [1., 0.], [0., 1.]],
    ])
    l, cw, ca = pairwise_stats(s)
    # row 0 pairs: (1,0)&(0,1): cos 0 -> 2-2*0 = 2; (1,0)&(0,-1): 0 -> 2;
    #              (0,1)&(0,-1): -1 -> 4  => mean 8/3
    # row 1 pairs: cos 1 -> 0; cos 0 -> 2; cos 0 -> 2 => mean 4/3
    expect_l = (8 / 3 + 4 / 3) / 2
    expect_cw = (0 + 0 - 1 + 1 + 0 + 0) / 6
    assert abs(l - expect_l) < 1e-6, (l, expect_l)
    assert abs(cw - expect_cw) < 1e-6, (cw, expect_cw)
    # cos_across: per slot, across the two states.
    # slot 0: (1,0)~ (1,0) = 1; slot 1: (0,1)~(1,0) = 0; slot 2: (0,-1)~(0,1) = -1
    expect_ca = (1 + 0 + (-1)) / 3
    assert abs(ca - expect_ca) < 1e-6, (ca, expect_ca)
    print(f'  hand-computed stats: L {l:.4f} cos_within {cw:.4f} '
          f'cos_across {ca:.4f}')

    # the constant map: every cosine -> 1, L -> 0
    const = torch.ones(4, 5, 3)
    const = torch.nn.functional.normalize(const, dim=-1)
    l0, cw0, ca0 = pairwise_stats(const)
    assert abs(l0) < 1e-6 and abs(cw0 - 1.0) < 1e-6 and abs(ca0 - 1.0) < 1e-6, (
        l0, cw0, ca0)
    print('  constant map: L 0.0, both cosines 1.0')
    # S=1 must not divide by zero in the across-state term -- the readout is
    # only ever run on S=128, but the fixture states the boundary explicitly:
    # with S=1 there are no across-state pairs, so the caller would see NaN and
    # the guard belongs at the tool level, not here. (Documented, not asserted.)
    print('  (S=1 across-state is undefined -- the tool runs S=128 only)')


def test_collapse_signature():
    ok = collapse_signature((0.01, 0.9995, 0.9998), (0.5, 0.5, 0.1))
    assert ok is True
    # a healthy head: low within-cosine is not collapse even with tiny L
    assert collapse_signature((0.01, 0.5, 0.1), (0.5, 0.5, 0.1)) is False
    # a constant-ish head that scored near the control's L is NOT the signature
    assert collapse_signature((0.49, 0.9995, 0.9998), (0.5, 0.5, 0.1)) is False
    # view-invariant but state-preserving: high cos_within, low cos_across
    assert collapse_signature((0.05, 0.999, 0.2), (0.5, 0.5, 0.1)) is False
    print('  collapse predicate separates constant-map from view-invariant')


def test_latest_npz():
    with tempfile.TemporaryDirectory() as tmp:
        for ep in (0, 25, 200):
            # via a handle: np.save appends .npy to a bare string path
            with open(os.path.join(tmp, f'epoch_{ep:04d}.npz'), 'wb') as f:
                np.save(f, np.zeros(1))
        p = latest_npz(tmp)
        assert p.endswith('epoch_0200.npz'), p
        q = latest_npz(p)          # a file passes through
        assert q == p
        empty = os.path.join(tmp, 'empty')
        os.makedirs(empty)
        try:
            latest_npz(empty)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('empty dir accepted')
    print('  latest_npz picks the last epoch and passes files through')


def test():
    print('--- analyze_semantic_head ---')
    test_pairwise_stats()
    test_collapse_signature()
    test_latest_npz()
    print('ALL SEMANTIC-READOUT CHECKS PASSED')


if __name__ == '__main__':
    test()
