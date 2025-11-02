from pathlib import Path
import os
import argparse

import torch
import numpy as np
import trimesh
from tqdm import tqdm
import open3d as o3d

# ---- your existing utils (we rely on their behavior) ----
# mask_depth_to_pts returns camera-frame points from (mask, depth, K)
# project_points expects OpenCV-style W2C extrinsics and K
from dataset.database import (
    parse_database_name, get_database_split,
    GlossySyntheticDatabase, NeRFSyntheticDatabase, get_database_eval_points
)
from utils.base_utils import (
    mask_depth_to_pts, project_points, pose_apply, pose_inverse # pose_apply -> transform_points_pose under the hood
)
# NOTE: We do NOT use your old pose_inverse here for NeRF; we compute w2c from c2w via inv+S.

# -------------------------
# Convention bridge helpers
# -------------------------

def nearest_dist(pts0, pts1, batch_size=512):
    pts0 = torch.from_numpy(pts0.astype(np.float32)).cuda()
    pts1 = torch.from_numpy(pts1.astype(np.float32)).cuda()
    pn0, pn1 = pts0.shape[0], pts1.shape[0]
    dists = []
    for i in tqdm(range(0, pn0, batch_size), desc='evaluting...'):
        dist = torch.norm(pts0[i:i+batch_size,None,:] - pts1[None,:,:], dim=-1)
        dists.append(torch.min(dist,1)[0])
    dists = torch.cat(dists,0)
    return dists.cpu().numpy()

def to_4x4(M):
    M = np.asarray(M, dtype=np.float32)
    if M.shape == (4,4): return M
    if M.shape == (3,4): return np.vstack([M, np.array([[0,0,0,1]], np.float32)])
    raise ValueError(f"pose must be (3,4) or (4,4), got {M.shape}")

S4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)  # OpenGL cam -> OpenCV cam (and vice versa; S4 == S4^{-1})

def gl_c2w_to_cv_w2c(c2w_gl_4):
    # W2C_cv = S4 @ inv(C2W_gl)
    return S4 @ np.linalg.inv(c2w_gl_4)

def transform_points_col(pts_N3, T4x4):
    """Column-vector convention: X' = T @ X. pts:(N,3) -> (N,3)."""
    pts = np.asarray(pts_N3, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    ph = np.concatenate([pts, ones], axis=1)            # (N,4)
    wh = (T4x4 @ ph.T).T                                # (N,4)
    w = wh[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1.0, w)
    return (wh[:, :3] / w).astype(np.float32)

# ---------- your OpenCV-style rasterizer kept the same ----------

def rasterize_depth_map(mesh, pose_w2c_cv, K, shape, near=5e-1, far=1e2):
    import nvdiffrast.torch as dr
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # Ensure (3,4) W2C for your project_points()
    M = np.asarray(pose_w2c_cv, dtype=np.float32)
    if M.shape == (4,4): M = M[:3,:]
    assert M.shape == (3,4)

    from utils.base_utils import project_points  # your existing OpenCV projection
    pts, depth = project_points(vertices, M, K)

    h, w = shape
    pts[:,0] = (pts[:,0]*2 - w) / w
    pts[:,1] = (pts[:,1]*2 - h) / h

    z = (depth - near) / (far - near)
    z = z*2 - 1
    pts_clip = np.concatenate([pts, z[:,None]], 1)

    pts_clip = torch.from_numpy(pts_clip.astype(np.float32)).cuda()
    indices = torch.from_numpy(faces.astype(np.int32)).cuda()
    pts_clip = torch.cat([pts_clip, torch.ones_like(pts_clip[...,0:1])], 1).unsqueeze(0)

    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, pts_clip, indices, (h, w))
    depth_img = (rast[0,:,:,2]+1)/2*(far-near)+near
    mask = rast[0,:,:,-1] != 0
    return depth_img.detach().cpu().numpy(), mask.detach().cpu().numpy().astype(bool)

# ---------- build world-space point sets for NeRF Synthetic ----------



def get_mesh_eval_points_nerf(database, mesh_path):
    """
    Predicted mesh samples in world coords via rasterized depth per view.

    For each view:
      c2w_gl
      -> w2c_cv = S4 @ inv(c2w_gl)                       (OpenCV W2C for rasterizer)
      -> depth_pr, mask_pr = rasterize_depth_map(mesh, w2c_cv, K, (H,W))
      -> pts_cam_cv  = mask_depth_to_pts(mask_pr, depth_pr, K)
      -> T_camcv2w   = C2W_gl @ S4
      -> pts_world   = transform_points_col(pts_cam_cv, T_camcv2w)
    """
    from dataset.database import get_database_split, NeRFSyntheticDatabase
    from utils.base_utils import mask_depth_to_pts
    assert isinstance(database, NeRFSyntheticDatabase)

    mesh = trimesh.load_mesh(mesh_path)
    _, _, test_ids = get_database_split(database, 'test')

    all_pts = []
    for img_id in tqdm(test_ids, desc='Mesh->world'):
        K = database.get_K(img_id)
        c2w_gl = to_4x4(database.get_pose(img_id))
        H, W, _ = database.get_image(img_id).shape

        w2c_cv = gl_c2w_to_cv_w2c(c2w_gl)                  # (4,4)
        depth_pr, mask_pr = rasterize_depth_map(mesh, w2c_cv, K, (H, W))

        pts_cam_cv = mask_depth_to_pts(mask_pr, depth_pr, K)
        # pts_cam_cv[:, 0] = -pts_cam_cv[:, 0]  # OpenCV to OpenGL cam x-flip
        T_camcv2w = c2w_gl @ S4
        pts_world = transform_points_col(pts_cam_cv, T_camcv2w)
        if pts_world.size:
            all_pts.append(pts_world)

    if not all_pts:
        return np.zeros((0,3), np.float32)

    pts = np.concatenate(all_pts, 0).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    down = pcd.voxel_down_sample(voxel_size=0.01)
    o3d.io.write_point_cloud("data/eval.ply", down)
    return np.asarray(down.points, np.float32)

# -------------------------
# Main script (NeRF Synthetic)
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, required=True)
    parser.add_argument('--object', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    # Build NeRF Synthetic DB
    database = parse_database_name(f'nerf/{args.object}', 'data/nerf_synthetic')

    # GT and predicted point clouds in WORLD coordinates
    pts_gt = get_database_eval_points(database)
    pts_pr = get_mesh_eval_points_nerf(database, args.mesh)

    print(f"pts_gt min: {pts_gt.min():.5f}, max: {pts_gt.max():.5f}, shape: {pts_gt.shape}")
    print(f"pts_pr min: {pts_pr.min():.5f}, max: {pts_pr.max():.5f}, shape: {pts_pr.shape}")

    # Chamfer
    dist_gt = nearest_dist(pts_gt, pts_pr, args.batch_size)
    dist_pr = nearest_dist(pts_pr, pts_gt, args.batch_size)
    chamfer = 0.5 * (np.mean(dist_gt) + np.mean(dist_pr))

    stem = Path(args.mesh).stem
    results = f'{stem} {chamfer:.5f}'
    print(results)
    os.makedirs('data', exist_ok=True)
    with open('data/geometry_nerf.log', 'a') as f:
        f.write(results + '\n')

if __name__ == "__main__":
    main()
