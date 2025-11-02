import os
import torch
import numpy as np
import argparse
import trimesh
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

from NeRO.dataset.database import get_database_eval_points_gem
# Assuming these imports are in your environment
from dataset.database import parse_database_name, get_database_split, GlossySyntheticDatabase, \
    NeRFSyntheticDatabase, get_database_eval_points_gem  # Added NeRFSyntheticDatabase
from utils.base_utils import color_map_backward  # Removed unused imports

# --- Coordinate System Transformation ---
# This matrix transforms points from NeRF/OpenGL camera space (x right, y up, z backward)
# to OpenCV camera space (x right, y down, z forward)
T_cv_gl = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

# Its inverse (OpenGL -> OpenCV) is conveniently the same matrix
T_gl_cv = T_cv_gl


# --- Original Helper Functions ---
# (from your second code block)

def mask_depth_to_pts(mask, depth, K, rgb=None):
    """
    Unprojects depth points to 3D in OpenCV camera space (+z forward).
    """
    hs, ws = np.nonzero(mask)
    depth = depth[hs, ws]
    pts = np.asarray([ws, hs, depth], np.float32).transpose()
    pts[:, :2] *= pts[:, 2:]
    if rgb is not None:
        return np.dot(pts, np.linalg.inv(K).transpose()), rgb[hs, ws]
    else:
        return np.dot(pts, np.linalg.inv(K).transpose())


def project_points(pts, RT, K):
    """
    Projects 3D points (world) to 2D image plane using OpenCV w2c pose (3x4).
    """
    pts = np.matmul(pts, RT[:, :3].transpose()) + RT[:, 3:].transpose()
    pts = np.matmul(pts, K.transpose())
    dpt = pts[:, 2]
    mask0 = (np.abs(dpt) < 1e-4) & (np.abs(dpt) > 0)
    if np.sum(mask0) > 0: dpt[mask0] = 1e-4
    mask1 = (np.abs(dpt) > -1e-4) & (np.abs(dpt) < 0)
    if np.sum(mask1) > 0: dpt[mask1] = -1e-4
    pts2d = pts[:, :2] / dpt[:, None]
    return pts2d, dpt


def pose_inverse(pose):
    """
    Inverts a 3x4 OpenCV w2c pose.
    """
    R = pose[:, :3].T
    t = - R @ pose[:, 3:]
    return np.concatenate([R, t], -1)


def pose_apply(pose, pts):
    """
    Applies a 3x4 pose to 3D points.
    """
    R = pose[:, :3]
    t = pose[:, 3:]
    return np.matmul(pts, R.T) + t.T  # Corrected from your provided util


# --- New Helper Functions for 4x4 Poses ---

def pose_inverse_4x4(pose):
    """
    Inverts a 4x4 pose.
    """
    return np.linalg.inv(pose)


def pose_apply_4x4(pose, pts):
    """
    Applies a 4x4 pose to 3D points (N, 3).
    """
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)  # (N, 4)
    # (4, 4) @ (4, N) -> (4, N) -> (N, 4)
    pts_transformed_h = (pose @ pts_h.T).T
    # Normalize by w
    pts_transformed = pts_transformed_h[:, :3] / np.where(
        pts_transformed_h[:, 3:4] == 0, 1e-6, pts_transformed_h[:, 3:4]
    )
    return pts_transformed


# --- Core Evaluation Functions (Modified) ---

def nearest_dist(pts0, pts1, batch_size=512):
    pts0 = torch.from_numpy(pts0.astype(np.float32)).cuda()
    pts1 = torch.from_numpy(pts1.astype(np.float32)).cuda()
    pn0, pn1 = pts0.shape[0], pts1.shape[0]
    dists = []
    with torch.no_grad():
        for i in tqdm(range(0, pn0, batch_size), desc='evaluating...'):
            dist = torch.norm(pts0[i:i + batch_size, None, :] - pts1[None, :, :], dim=-1)
            dists.append(torch.min(dist, 1)[0])
    dists = torch.cat(dists, 0)
    return dists.cpu().numpy()


def rasterize_depth_map(mesh, pose, K, shape):
    """
    Rasterizes a mesh to a depth map using an OpenCV w2c pose (3x4).
    """
    try:
        import nvdiffrast.torch as dr
    except ImportError:
        print("Error: nvdiffrast not found. Please install it.")
        return None, None

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    # Project vertices using the OpenCV w2c pose
    pts, depth = project_points(vertices, pose, K)

    # normalize to projection
    h, w = shape
    pts[:, 0] = (pts[:, 0] * 2 - w) / w
    pts[:, 1] = (pts[:, 1] * 2 - h) / h
    near, far = 5e-1, 1e2  # Default near/far, adjust if needed
    z = (depth - near) / (far - near)
    z = z * 2 - 1
    pts_clip = np.concatenate([pts, z[:, None]], 1)

    pts_clip = torch.from_numpy(pts_clip.astype(np.float32)).cuda()
    indices = torch.from_numpy(faces.astype(np.int32)).cuda()
    pts_clip = torch.cat([pts_clip, torch.ones_like(pts_clip[..., 0:1])], 1).unsqueeze(0)

    # ctx = dr.RasterizeGLContext() # Use GL context
    ctx = dr.RasterizeCudaContext()  # Or Cuda context

    rast, _ = dr.rasterize(ctx, pts_clip, indices, (h, w))  # [1,h,w,4]
    depth = (rast[0, :, :, 2] + 1) / 2 * (far - near) + near
    mask = rast[0, :, :, -1] != 0
    return depth.cpu().numpy(), mask.cpu().numpy().astype(bool)


def get_mesh_eval_points(database, mesh_path):
    """
    Generates a point cloud from the mesh by rasterizing it from all test views.
    Handles both GlossySynthetic (OpenCV) and NeRFSynthetic (OpenGL).
    """
    mesh = trimesh.load_mesh(mesh_path)
    if not mesh:
        print(f"Error: Could not load mesh from {mesh_path}")
        return None

    _, test_ids, _ = get_database_split(database, 'test')
    pbar = tqdm(test_ids, desc="Rasterizing mesh")
    pts_pr = []

    for test_id in pbar:
        K = database.get_K(test_id)  # (3, 3)
        h, w, _ = database.get_image(test_id).shape

        if isinstance(database, NeRFSyntheticDatabase):
            # NeRF: pose is (4, 4) c2w, OpenGL convention
            pose_c2w_gl = database.get_pose(test_id)

            # 1. Convert to (3, 4) w2c, OpenCV convention for rasterizer
            pose_w2c_gl = pose_inverse_4x4(pose_c2w_gl)
            pose_w2c_cv_4x4 = T_cv_gl @ pose_w2c_gl
            pose_w2c_cv_3x4 = pose_w2c_cv_4x4[:3, :]

            # 2. Rasterize
            depth_pr, mask_pr = rasterize_depth_map(mesh, pose_w2c_cv_3x4, K, (h, w))
            if depth_pr is None: continue

            # 3. Unproject points (to OpenCV camera space)
            pts_cv = mask_depth_to_pts(mask_pr, depth_pr, K)  # (N, 3)

            # 4. Transform points to world space
            # OpenCV_cam -> OpenGL_cam -> World
            pts_gl = pose_apply_4x4(T_gl_cv, pts_cv)
            pts_world = pose_apply_4x4(pose_c2w_gl, pts_gl)
            pts_pr.append(pts_world)

        elif isinstance(database, GlossySyntheticDatabase):
            # Glossy: pose is (3, 4) w2c, OpenCV convention
            pose_w2c_cv = database.get_pose(test_id)  # (3, 4)

            # 1. Rasterize directly
            depth_pr, mask_pr = rasterize_depth_map(mesh, pose_w2c_cv, K, (h, w))  # (H, W)
            if depth_pr is None: continue

            # 2. Unproject points (to OpenCV camera space)
            pts_cv = mask_depth_to_pts(mask_pr, depth_pr, K)  # (N, 3)

            # 3. Transform points to world space
            # Get (3, 4) c2w pose
            pose_c2w_cv = pose_inverse(pose_w2c_cv)  # (3,4)
            pts_world = pose_apply(pose_c2w_cv, pts_cv)  # (N, 3)
            pts_pr.append(pts_world)

        else:
            raise NotImplementedError(f"Database type {type(database)} not supported.")

    if not pts_pr:
        print("Error: No points generated from mesh.")
        return None

    pts_pr = np.concatenate(pts_pr, 0).astype(np.float32)
    print(f"Mesh points before downsample: {pts_pr.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_pr)
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)  # Voxel size as in original
    print(f"Mesh points after downsample: {len(downpcd.points)}")
    return np.asarray(downpcd.points, np.float32)



# --- Main Execution ---

def main(args):
    # --- IMPORTANT ---
    # Update this section based on how your parse_database_name works
    # This example assumes it can handle 'nerf_synthetic/lego' or 'syn/drum'

    if args.dataset_type == 'nerf':
        # Example for NeRF Synthetic
        database_name = f'nerf_synthetic/{args.object}'
        database_root = 'data/nerf_synthetic'  # Adjust path
    elif args.dataset_type == 'glossy':
        # Example for Glossy Synthetic
        database_name = f'syn/{args.object}'
        database_root = 'data/GlossySynthetic'  # Adjust path
    else:
        print(f"Error: Unknown dataset_type '{args.dataset_type}'")
        return

    # This function must return the correct Database object
    # (e.g., NeRFSyntheticDatabase or GlossySyntheticDatabase)
    database = parse_database_name(database_name, database_root)
    if database is None:
        print(f"Error: Could not parse database {database_name}")
        return

    # 1. Get Ground Truth points
    pts_gt = get_database_eval_points_gem(database)
    if pts_gt is None or pts_gt.shape[0] == 0:
        print("Error: Failed to get ground truth points.")
        return

    # 2. Get Predicted Mesh points
    pts_pr = get_mesh_eval_points(database, args.mesh)
    if pts_pr is None or pts_pr.shape[0] == 0:
        print("Error: Failed to get predicted mesh points.")
        return

    # 3. Calculate Chamfer Distance
    print("Calculating GT -> PR distance...")
    dist_gt = nearest_dist(pts_gt, pts_pr, args.batch_size)
    print("Calculating PR -> GT distance...")
    dist_pr = nearest_dist(pts_pr, pts_gt, args.batch_size)

    dist_gt_mean = np.mean(dist_gt)
    dist_pr_mean = np.mean(dist_pr)
    chamfer = (dist_gt_mean + dist_pr_mean) / 2

    stem = Path(args.mesh).stem
    results = f'Object: {args.object}, Mesh: {stem}, Chamfer: {chamfer:.6f}, GT->PR: {dist_gt_mean:.6f}, PR->GT: {dist_pr_mean:.6f}'
    print(results)

    log_file = 'data/geometry.log'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(results + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, required=True, help="Path to the reconstructed .ply or .obj mesh.")
    parser.add_argument('--object', type=str, required=True, help="Object name (e.g., 'lego', 'drum').")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['nerf', 'glossy'],
                        help="Type of dataset (nerf or glossy).")
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: This script requires a GPU and CUDA.")
    else:
        main(args)
