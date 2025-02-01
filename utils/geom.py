import os
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError
import torch

from utils import math as mathutil

def write_lvis(lvis, fps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Dump raw
    raw_out = os.path.join(out_dir, 'lvis.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, lvis)
    # Visualize the average across all lights as an image
    vis_out = os.path.join(out_dir, 'lvis.png')
    lvis_avg = np.mean(lvis, axis=2)
    # Replace xm.io.img.write_arr with appropriate visualization code
    # Placeholder for writing the image
    np.save(vis_out, lvis_avg)
    # Visualize light visibility for each light pixel
    vis_out = os.path.join(out_dir, 'lvis.mp4')
    frames = []
    for i in range(lvis.shape[2]):  # for each light pixel
        frame = lvis[:, :, i]  # Normalize and stack as needed
        frame = np.dstack([frame] * 3)
        frames.append(frame)
    # Replace xm.vis.video.make_video with appropriate video creation code
    # Placeholder for creating a video
    np.save(vis_out, frames)

def write_xyz(xyz_arr, out_dir):
    arr = xyz_arr
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    # Dump raw
    raw_out = os.path.join(out_dir, 'xyz.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization
    vis_out = os.path.join(out_dir, 'xyz.png')
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    # Replace xm.io.img.write_arr with appropriate visualization code
    # Placeholder for writing the image
    np.save(vis_out, arr_norm)

def write_normal(arr, out_dir):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    # Dump raw
    raw_out = os.path.join(out_dir, 'normal.npy')
    with open(raw_out, 'wb') as h:
        np.save(h, arr)
    # Visualization
    vis_out = os.path.join(out_dir, 'normal.png')
    arr = (arr + 1) / 2
    # Replace xm.io.img.write_arr with appropriate visualization code
    # Placeholder for writing the image
    np.save(vis_out, arr)

def write_alpha(arr, out_dir):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    vis_out = os.path.join(out_dir, 'alpha.png')
    # Replace xm.io.img.write_arr with appropriate visualization code
    # Placeholder for writing the image
    np.save(vis_out, arr)

def get_convex_hull(pts):
    try:
        hull = ConvexHull(pts)
    except QhullError:
        hull = None
    return hull

def in_hull(hull, pts):
    verts = hull.points[hull.vertices, :]
    hull = Delaunay(verts)
    return hull.find_simplex(pts) >= 0

def rad2deg(rad):
    return 180 / np.pi * rad

def slerp(p0, p1, t):
    assert p0.ndim == p1.ndim == 2, "Vectors must be 2D"

    if p0.shape[0] == 1:
        cos_omega = p0 @ p1.T
    elif p0.shape[1] == 1:
        cos_omega = p0.T @ p1
    else:
        raise ValueError("Vectors should have one singleton dimension")

    omega = mathutil.safe_acos(cos_omega)

    z0 = p0 * torch.sin((1 - t) * omega) / torch.sin(omega)
    z1 = p1 * torch.sin(t * omega) / torch.sin(omega)

    z = z0 + z1
    return z

def gen_world2local(normal, eps=1e-6):
    """Generates rotation matrices that transform world normals to local +Z
    (world tangents to local +X, and world binormals to local +Y).

    `normal`: Nx3
    """
    normal = mathutil.safe_l2_normalize(normal, dim=1)

    # To avoid colinearity with some special normals that may pop up
    z = torch.tensor([0, 0, 1], dtype=torch.float32, device=normal.device) + eps
    z = z.unsqueeze(0).repeat(normal.shape[0], 1)

    # Tangents
    t = torch.cross(normal, z, dim=1)
    assert torch.all(torch.linalg.norm(t, dim=1) > 0), (
        "Found zero-norm tangents, either because of colinearity "
        "or zero-norm normals")
    t = mathutil.safe_l2_normalize(t, dim=1)

    # Binormals
    b = torch.cross(normal, t, dim=1)
    b = mathutil.safe_l2_normalize(b, dim=1)

    # Rotation matrices
    rot = torch.stack((t, b, normal), dim=1)
    # So that at each pixel, we have a 3x3 matrix whose ROWS are world
    # tangents, binormals, and normals

    return rot

def dir2rusink(a, b):
    """Adapted from
    third_party/nielsen2015on/coordinateFunctions.py->DirectionsToRusink().

    `a` and `b` should be both Nx3.
    """
    a = mathutil.safe_l2_normalize(a, dim=1)
    b = mathutil.safe_l2_normalize(b, dim=1)
    h = mathutil.safe_l2_normalize((a + b) / 2, dim=1)

    theta_h = mathutil.safe_acos(h[:, 2])
    phi_h = mathutil.safe_atan2(h[:, 1], h[:, 0])

    binormal = torch.tensor((0, 1, 0), dtype=torch.float32, device=a.device)
    normal = torch.tensor((0, 0, 1), dtype=torch.float32, device=a.device)

    def rot_vec(vector, axis, angle):
        """Rotates vector around arbitrary axis.
        """
        cos_ang = torch.cos(angle).view(-1)
        sin_ang = torch.sin(angle).view(-1)
        vector = vector.view(-1, 3)
        axis = axis.view(-1, 3)
        return vector * cos_ang[:, None] + \
               axis * torch.matmul(vector, axis.T).diag().view(-1, 1) * (1 - cos_ang[:, None]) + \
               torch.cross(axis.expand_as(vector), vector) * sin_ang[:, None]

    # What is the incoming/outgoing direction in the Rusink. frame?
    diff = rot_vec(rot_vec(b, normal, -phi_h), binormal, -theta_h)
    diff0, diff1, diff2 = diff[:, 0], diff[:, 1], diff[:, 2]
    theta_d = mathutil.safe_acos(diff2)
    phi_d = torch.fmod(mathutil.safe_atan2(diff1, diff0), np.pi)
    rusink = torch.stack((phi_d, theta_h, theta_d), dim=1)

    return rusink
