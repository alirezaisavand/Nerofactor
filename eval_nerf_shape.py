from pathlib import Path
import os
import argparse

import torch
import numpy as np
import trimesh
from skimage.io import imsave
from tqdm import tqdm

from dataset.database import parse_database_name, get_database_split, get_database_eval_points, GlossySyntheticDatabase, NeRFSyntheticDatabase
from utils.base_utils import mask_depth_to_pts, project_points, color_map_backward, pose_inverse, pose_apply

import open3d as o3d


# ------------------------
# Utilities (OpenGL/Blender)
# ------------------------

def project_points_opengl(vertices_world: np.ndarray,
                          c2w: np.ndarray,
                          K: np.ndarray,
                          img_shape: tuple):
    """
    Project world-space vertices to image pixels (u,v) using OpenGL/Blender convention.

    Args:
        vertices_world: (N,3) float32
        c2w: (4,4) camera-to-world (Blender/NeRF synthetic)
        K: (3,3) intrinsics (fx, fy, cx, cy)
        img_shape: (H, W)

    Returns:
        pts_px: (N,2) in pixel coords (u right, v down)
        depth_lin: (N,) positive distance in front of camera (== -Z_cam)
        valid: (N,) boolean for depth > 0
    """
    H, W = img_shape
    c2w = to_4x4(c2w)
    w2c = np.linalg.inv(c2w).astype(np.float32)

    v_h = np.concatenate([vertices_world.astype(np.float32), np.ones((vertices_world.shape[0], 1), np.float32)], axis=1)
    vc = (w2c @ v_h.T).T[:, :3]  # (N,3) camera coords (OpenGL: camera looks along -Z)

    depth_lin = -vc[:, 2]                   # positive in front of camera
    valid = depth_lin > 0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # pinhole to pixels (image v grows downward)
    x_over_z = vc[:, 0] / (-vc[:, 2] + 1e-8)
    y_over_z = vc[:, 1] / (-vc[:, 2] + 1e-8)
    u = fx * x_over_z + cx
    v = fy * y_over_z + cy

    pts_px = np.stack([u, v], axis=1)
    return pts_px, depth_lin, valid


def rasterize_depth_map(mesh, pose_c2w, K, shape, near=5e-1, far=1e2):
    """
    Rasterize depth with OpenGL/Blender (NeRF synthetic) convention.

    - Projects world verts -> pixels with OpenGL (y up).
    - Converts pixels to OpenGL NDC: x_ndc = 2u/W - 1, y_ndc = 1 - 2v/H.
    - Maps *linear* depth (near..far) into z_ndc in [-1,1] for nvdiffrast.

    Returns:
        depth: (H,W) linear depth in [near, far]
        mask:  (H,W) bool
    """
    import nvdiffrast.torch as dr
    pose_c2w = to_4x4(pose_c2w)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    H, W = shape

    # Project using OpenGL convention
    pts_px, depth_lin, valid = project_points_opengl(vertices, pose_c2w, K, (H, W))

    # Pixel -> OpenGL NDC (y up). Your original code had no flip (OpenCV style)
    x_ndc = (pts_px[:, 0] / W) * 2.0 - 1.0
    y_ndc = 1.0 - (pts_px[:, 1] / H) * 2.0

    # Linear depth -> ndc z \in [-1,1] (not true GL z; we invert later the same way)
    z_ndc = ((depth_lin - near) / (far - near)) * 2.0 - 1.0

    pts_clip = np.stack([x_ndc, y_ndc, z_ndc], axis=1).astype(np.float32)

    pts_clip = torch.from_numpy(pts_clip).cuda()
    indices = torch.from_numpy(faces.astype(np.int32)).cuda()
    pts_clip = torch.cat([pts_clip, torch.ones_like(pts_clip[..., 0:1])], dim=1).unsqueeze(0)

    # Rasterize (GL-style NDC expected)
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, pts_clip, indices, (H, W))  # [1,H,W,4]

    # Recover linear depth from z_ndc
    depth = (rast[0, :, :, 2] + 1.0) * 0.5 * (far - near) + near
    mask = rast[0, :, :, -1] != 0

    return depth.detach().cpu().numpy(), mask.detach().cpu().numpy().astype(bool)


# ------------------------
# Distance utility (unchanged)
# ------------------------

def nearest_dist(pts0, pts1, batch_size=512):
    pts0 = torch.from_numpy(pts0.astype(np.float32)).cuda()
    pts1 = torch.from_numpy(pts1.astype(np.float32)).cuda()
    pn0, pn1 = pts0.shape[0], pts1.shape[0]
    dists = []
    for i in tqdm(range(0, pn0, batch_size), desc='evaluating...'):
        dist = torch.norm(pts0[i:i + batch_size, None, :] - pts1[None, :, :], dim=-1)
        dists.append(torch.min(dist, 1)[0])
    dists = torch.cat(dists, 0)
    return dists.cpu().numpy()


# ------------------------
# Pose helpers
# ------------------------

# ------------------------
# Point set generators
# ------------------------

def get_mesh_eval_points(database):
    """
    For NeRFSyntheticDatabase:
      - database.get_pose(id): c2w (4x4)
      - Rasterize predicted mesh per view to get a depth map (camera frame),
        back-project to camera points, then transform to world via c2w.
    """
    if isinstance(database, NeRFSyntheticDatabase):
        _, _, test_ids = get_database_split(database, 'test')
        mesh = trimesh.load_mesh(args.mesh)
        pbar = tqdm(total=len(test_ids), desc='mesh->pts')
        pts_pr = []
        for test_id in test_ids:
            K = database.get_K(test_id)          # (3,3)
            c2w = database.get_pose(test_id)     # (4,4), camera-to-world
            c2w = to_4x4(c2w)
            H, W, _ = database.get_image(test_id).shape

            depth_pr, mask_pr = rasterize_depth_map(mesh, c2w, K, (H, W))
            pts_cam = mask_depth_to_pts(mask_pr, depth_pr, K)  # camera-frame points
            pts_world = pose_apply(c2w, pts_cam)               # camera->world (no inverse)
            pts_pr.append(pts_world)
            pbar.update(1)

        pts_pr = np.concatenate(pts_pr, axis=0).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        print(pts_pr.shape, pts_pr.dtype)
        pcd.points = o3d.utility.Vector3dVector(pts_pr)
        downpcd = pcd.voxel_down_sample(voxel_size=0.01)
        return np.asarray(downpcd.points, np.float32)

    elif isinstance(database, GlossySyntheticDatabase):
        # Use your original OpenCV-convention pipeline for GlossySynthetic if needed.
        raise NotImplementedError("For GlossySyntheticDatabase, use the OpenCV-convention version.")
    else:
        raise NotImplementedError

# --- add this near the top (with other helpers) ---

def to_4x4(pose):
    """
    Normalize pose to a homogeneous 4x4 matrix (row-major).
    Accepts:
      - (4,4) -> returned as-is
      - (3,4) -> append [0,0,0,1]
      - (12,) or (16,) -> reshaped to (3,4) or (4,4) respectively (row-major)
    """
    pose = np.asarray(pose).astype(np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.vstack([pose, bottom])
    if pose.shape == (12,):
        pose34 = pose.reshape(3, 4)
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        return np.vstack([pose34, bottom])
    if pose.shape == (16,):
        return pose.reshape(4, 4)
    raise ValueError(f"Unsupported pose shape {pose.shape}; expected (3,4), (4,4), (12,), or (16,)")


def _as_o3d_points(pts_list_or_array):
    """
    Accepts a list of arrays or a single array of points.
    Returns a contiguous (N,3) float64 ndarray suitable for Open3D.
    """
    if isinstance(pts_list_or_array, list):
        if len(pts_list_or_array) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        pts = np.concatenate(pts_list_or_array, axis=0)
    else:
        pts = np.asarray(pts_list_or_array)

    # Fix common shape issues: (3, N) -> (N, 3)
    if pts.ndim != 2:
        raise ValueError(f"Points must be 2D, got shape {pts.shape}")
    if pts.shape[1] != 3 and pts.shape[0] == 3:
        pts = pts.T
    if pts.shape[1] != 3:
        raise ValueError(f"Points must have shape (N,3), got {pts.shape}")

    # Open3D is happy with float64
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    return pts



# ------------------------
# Main
# ------------------------

def main():
    database = parse_database_name(f'nerf/{args.object}', 'data/nerf_synthetic')
    pts_gt = get_database_eval_points(database)
    pts_pr = get_mesh_eval_points(database)

    print(f"pts_gt min: {pts_gt.min()}, max: {pts_gt.max()}, num: {pts_gt.shape}")
    print(f"pts_pr min: {pts_pr.min()}, max: {pts_pr.max()}, num: {pts_pr.shape}")

    dist_gt = nearest_dist(pts_gt, pts_pr, args.batch_size)
    dist_pr = nearest_dist(pts_pr, pts_gt, args.batch_size)

    stem = Path(args.mesh).stem
    chamfer = (np.mean(dist_gt) + np.mean(dist_pr)) / 2
    results = f'{stem} {chamfer:.5f}'
    print(results)
    os.makedirs('data', exist_ok=True)
    with open('data/geometry.log', 'a') as f:
        f.write(results + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--object', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    main()
