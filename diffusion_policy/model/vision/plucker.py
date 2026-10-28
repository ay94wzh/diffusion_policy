"""Pluecker ray maps for mujoco cameras (torch only -- no simulator dependency).

Convention
----------
For pixel (row=i, col=j) the ray is taken at the pixel *centre*,
``(u, v) = (j + 0.5, i + 0.5)``, and mujoco cameras look along **-z** with
**+x right** and **+y up** in the camera frame::

    d_cam   = normalize([ (u - cx)/f,  -(v - cy)/f,  -1 ])     # note both minus signs
    R       = quat_wxyz_to_mat(q)                              # camera -> world
    d_world = R @ d_cam                                        # (..., 3, H, W)
    m_world = cam_pos x d_world                                # (..., 3, H, W)
    map     = cat([d_world, m_world], dim=-3)                  # (..., 6, H, W)

with ``f = (H/2) / tan(fovy_deg/2)`` and ``cx, cy = W/2, H/2`` -- the same
intrinsics as ``multiview_image_dataset.fovy_to_intrinsics``.

Row 0 is the **top** of the stored image: robomimic flips mujoco's bottom-up
``readPixels`` exactly once (``EnvRobosuite.get_observation``), and both the
hdf5 and the generated zarr store that flipped form. This is the same
orientation ``project_world_to_pixel`` assumes, and the two are numerically
inverses (see the test: max 1.4e-14 px over 200 random pixels).

Why world-frame moments
-----------------------
``d`` depends on the camera's *orientation only* -- two cameras with the same
rotation but different positions have bit-identical direction maps. ``m = o x d``
is therefore the channel that carries camera **translation**, which is exactly
what a novel-view policy needs. Moments taken in the camera frame are
identically zero (the origin is 0 there) and would be useless. ``m`` is also
invariant to sliding ``o`` along the ray (``(o + t d) x d == o x d``), which is
what makes the (d, m) pair a well-defined encoding of a *line* rather than of a
point on it.
"""
from typing import Optional

import torch
from torch import Tensor


def quat_wxyz_to_mat_torch(q: Tensor) -> Tensor:
    """(..., 4) unit quaternion in wxyz order -> (..., 3, 3) rotation matrix.

    Camera frame -> world, matching ``quat_wxyz_to_mat`` in
    ``multiview_image_dataset`` (whose convention M2's gate 2 validated against
    the simulator).

    Never reshapes: the components are indexed on the last dim and the 3x3 is
    built by stacking, so a 1-D ``(4,)`` quaternion yields a 2-D ``(3, 3)`` and
    a ``(V, 4)`` yields ``(V, 3, 3)``. An earlier draft used
    ``einsum('bij,jhw->bhwi', ...)``, which raised "number of subscripts in the
    equation (3) does not match the number of dimensions (2)" on a 1-D
    quaternion; there is deliberately no einsum or reshape here so that failure
    mode cannot recur.
    """
    if q.shape[-1] != 4:
        raise ValueError(
            f'expected a (..., 4) wxyz quaternion, got {tuple(q.shape)}')
    # no keepdim: the components come out of unbind with the leading dims only,
    # so the norm must broadcast against them (keepdim would make a (4,) input
    # produce (1,)-shaped components and a (3, 3, 1) matrix)
    norm = q.norm(dim=-1)
    if not bool((norm > 1e-6).all()):
        raise ValueError(
            'degenerate quaternion (norm <= 1e-6); a zero quaternion silently '
            'produces a zero rotation matrix and NaN ray directions')
    w, x, y, z = (c / norm for c in q.unbind(dim=-1))
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


def ray_dirs_cam(fovy_deg, h: int, w: int,
                 dtype: Optional[torch.dtype] = None,
                 device=None) -> Tensor:
    """Unit ray directions in the camera frame, ``(..., 3, h, w)``.

    ``fovy_deg`` may be a scalar (-> ``(3, h, w)``) or any leading shape
    (-> ``(..., 3, h, w)``), so per-view fovys work without a Python loop.
    """
    if dtype is None:
        dtype = torch.float32
    fovy = torch.as_tensor(fovy_deg, dtype=dtype, device=device)
    f = (h / 2.0) / torch.tan(torch.deg2rad(fovy) / 2.0)      # (...)
    inv_f = 1.0 / f                                            # (...)

    # pixel centres relative to the principal point, at focal length 1
    u = torch.arange(w, dtype=dtype, device=device) + 0.5 - w / 2.0
    v = torch.arange(h, dtype=dtype, device=device) + 0.5 - h / 2.0
    # +x right, +y up (so v is negated), -z forward
    grid = torch.stack([
        u.view(1, w).expand(h, w),
        -v.view(h, 1).expand(h, w),
        torch.full((h, w), -1.0, dtype=dtype, device=device),
    ], dim=0)                                                  # (3, h, w)

    # scale the x,y channels by 1/f; z stays -1. Built by stacking (cat needs
    # >=1-D, and a scalar fovy gives a 0-dim 1/f) then restored to (...)
    inv_f_flat = inv_f.reshape(-1)                              # (1,) or (V,)
    scale = torch.stack([inv_f_flat, inv_f_flat,
                         torch.ones_like(inv_f_flat)], dim=-1)   # (1,3) or (V,3)
    scale = scale.reshape(*f.shape, 3)
    d = grid * scale[..., :, None, None]                        # (..., 3, h, w)
    return d / d.norm(dim=-3, keepdim=True).clamp_min(1e-12)


def ray_dirs_cam_at(u: Tensor, v: Tensor, fovy_deg, h: int, w: int,
                    dtype: Optional[torch.dtype] = None,
                    device=None) -> Tensor:
    """Unit ray directions at arbitrary pixel-centre coordinates, ``(..., 3, n)``.

    ``u``/``v`` are continuous coordinates in the **original** image frame
    (``h``, ``w`` set the intrinsics), using ``ray_dirs_cam``'s convention:
    pixel index ``j`` has centre ``u = j + 0.5``. Unlike ``ray_dirs_cam``, which
    always samples the full ``h x w`` grid, this evaluates any set of points --
    the R encoder uses it to put one ray at each layer3 feature cell's
    receptive-field centre without materialising the full map.

    Shapes: ``u``/``v`` ``(..., n)`` and identical; ``fovy_deg`` a scalar or
    broadcastable with their leading shape; the result is ``(..., 3, n)``.
    """
    if dtype is None:
        dtype = torch.float32
    if tuple(u.shape) != tuple(v.shape):
        raise ValueError(
            f'u and v must have identical shapes, got {tuple(u.shape)} and '
            f'{tuple(v.shape)}')
    if u.dim() == 0:
        raise ValueError(
            'u/v need a sample axis; pass (n,) or (..., n), not 0-dim scalars')
    fovy = torch.as_tensor(fovy_deg, dtype=dtype, device=device)
    u = torch.as_tensor(u, dtype=dtype, device=device)
    v = torch.as_tensor(v, dtype=dtype, device=device)
    f = (h / 2.0) / torch.tan(torch.deg2rad(fovy) / 2.0)        # (...)
    # +x right, +y up (so v is negated), -z forward -- same as ray_dirs_cam
    x = (u - w / 2.0) / f[..., None]
    y = -(v - h / 2.0) / f[..., None]
    d = torch.stack([x, y, torch.full_like(x, -1.0)], dim=-2)   # (..., 3, n)
    return d / d.norm(dim=-2, keepdim=True).clamp_min(1e-12)


def plucker_ray_map_at(cam_pos: Tensor, cam_quat_wxyz: Tensor, fovy_deg,
                       h: int, w: int, u: Tensor, v: Tensor) -> Tensor:
    """Pluecker embedding ``[d(3) | m(3)]`` at arbitrary pixel coordinates.

    Same convention and world-frame moments as ``plucker_ray_map`` (see the
    module docstring); ``u``/``v`` are continuous coordinates in the original
    ``h x w`` image (pixel index ``j`` has centre ``j + 0.5``). ``cam_pos`` is
    ``(..., 3)`` and ``cam_quat_wxyz`` ``(..., 4)``; ``u``/``v`` are
    ``(..., n)`` with the same leading shape. Returns ``(..., 6, n)``.
    """
    if cam_pos.shape[-1] != 3:
        raise ValueError(f'expected a (..., 3) position, got {tuple(cam_pos.shape)}')
    if cam_pos.shape[:-1] != cam_quat_wxyz.shape[:-1]:
        raise ValueError(
            f'cam_pos leading shape {tuple(cam_pos.shape[:-1])} does not match '
            f'cam_quat_wxyz {tuple(cam_quat_wxyz.shape[:-1])}')
    if tuple(u.shape[:-1]) != tuple(cam_pos.shape[:-1]):
        raise ValueError(
            f'u/v leading shape {tuple(u.shape[:-1])} does not match the camera '
            f'{tuple(cam_pos.shape[:-1])}; a per-view camera needs one set of '
            f'coordinates per view, not a shared grid')
    R = quat_wxyz_to_mat_torch(cam_quat_wxyz)                   # (..., 3, 3)
    d_cam = ray_dirs_cam_at(
        u, v, fovy_deg, h, w, dtype=R.dtype, device=R.device)   # (..., 3, n)
    d_world = torch.matmul(R, d_cam)
    m_world = torch.cross(cam_pos.unsqueeze(-1), d_world, dim=-2)
    return torch.cat([d_world, m_world], dim=-2)                # (..., 6, n)


def plucker_ray_map(cam_pos: Tensor, cam_quat_wxyz: Tensor, fovy_deg,
                    h: int, w: int) -> Tensor:
    """Pluecker embedding ``[d(3) | m(3)]`` of the camera's rays.

    ``cam_pos``: ``(..., 3)``, ``cam_quat_wxyz``: ``(..., 4)`` wxyz camera->world,
    ``fovy_deg``: scalar or ``(...)``. Returns ``(..., 6, h, w)``.

    Unbatched inputs give ``(6, h, w)`` and batched ones ``(V, 6, h, w)``.
    """
    if cam_pos.shape[-1] != 3:
        raise ValueError(f'expected a (..., 3) position, got {tuple(cam_pos.shape)}')
    if cam_pos.shape[:-1] != cam_quat_wxyz.shape[:-1]:
        raise ValueError(
            f'cam_pos leading shape {tuple(cam_pos.shape[:-1])} does not match '
            f'cam_quat_wxyz {tuple(cam_quat_wxyz.shape[:-1])}')

    R = quat_wxyz_to_mat_torch(cam_quat_wxyz)                   # (..., 3, 3)
    d_cam = ray_dirs_cam(fovy_deg, h, w, dtype=R.dtype, device=R.device)
    # flatten the pixel axes only, keeping any leading view dim: d_cam is
    # (..., 3, h*w), NOT (3, h*w). The channel axis is at -3, so the batch slice
    # is [:-3] -- slicing [:-2] would fold the channel into the batch for an
    # unbatched (3, h, w) input. R @ d_cam then broadcasts.
    d_world = torch.matmul(R, d_cam.reshape(*d_cam.shape[:-3], 3, h * w))
    # m = o x d, over the size-3 axis; o broadcasts against the pixel axis
    m_world = torch.cross(cam_pos.unsqueeze(-1), d_world, dim=-2)
    out = torch.cat([d_world, m_world], dim=-2)                 # (..., 6, h*w)
    return out.reshape(*out.shape[:-2], 6, h, w)
