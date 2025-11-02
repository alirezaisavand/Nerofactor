from pathlib import Path

import torch
import numpy as np
import argparse

import trimesh
from skimage.io import imsave
from tqdm import tqdm

from dataset.database import parse_database_name, get_database_split, get_database_eval_points, GlossySyntheticDatabase
from utils.base_utils import mask_depth_to_pts, project_points, color_map_backward, pose_inverse, pose_apply
import open3d as o3d


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
    # knn = KNN(1)
    # dists = []
    # for i in tqdm(range(0, pn0, batch_size), desc='evaluting...'):
    #     batch_size_ = pn1//20
    #     dist = []
    #     for k in range(0, pn1, batch_size_):
    #         dist_, _ = knn(pts1[None,k:k+batch_size_,:].permute(0,2,1), pts0[None,i:i+batch_size,:].permute(0,2,1))
    #         dist.append(dist_[0,0])
    #     # dist = torch.norm(pts0[i:i+batch_size,None,:] - pts1[None,:,:], dim=-1)
    #     dists.append(torch.min(torch.stack(dist, 1),dim=1)[0])
    # dists = torch.cat(dists,0)
    # return dists

def rasterize_depth_map(mesh,pose,K,shape):
    import nvdiffrast.torch as dr
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    pts, depth = project_points(vertices,pose,K)
    # normalize to projection
    h, w = shape
    pts[:,0]=(pts[:,0]*2-w)/w
    pts[:,1]=(pts[:,1]*2-h)/h
    near, far = 5e-1, 1e2
    z = (depth-near)/(far-near)
    z = z*2 - 1
    pts_clip = np.concatenate([pts,z[:,None]],1)

    pts_clip = torch.from_numpy(pts_clip.astype(np.float32)).cuda()
    indices = torch.from_numpy(faces.astype(np.int32)).cuda()
    pts_clip = torch.cat([pts_clip,torch.ones_like(pts_clip[...,0:1])],1).unsqueeze(0)
    # ctx = dr.RasterizeGLContext()
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, pts_clip, indices, (h, w)) # [1,h,w,4]
    depth = (rast[0,:,:,2]+1)/2*(far-near)+near
    mask = rast[0,:,:,-1]!=0
    return depth.cpu().numpy(), mask.cpu().numpy().astype(bool)

def gl_to_cv_camera(K_gl, pose_gl, pose_type="c2w"):
    """
    Convert camera intrinsics/extrinsics from OpenGL (Blender/NeRF) to OpenCV convention.

    Conventions
    ----------
    OpenGL/Blender camera coords : x right, y up,   z backward (camera looks along -Z)
    OpenCV camera coords         : x right, y down, z forward

    The fixed transform between camera frames is:
        X_cv = S * X_gl,  where  S = diag(1, -1, -1)

    Therefore, extrinsics convert as:
        If pose_gl is C2W (camera->world):   C2W_cv = C2W_gl * S
        If pose_gl is W2C (world->camera):   W2C_cv = S * W2C_gl

    Intrinsics K
    ------------
    K is defined in pixel coordinates (u right, v down). As long as you convert the
    camera frame via S as above, K does not need to change numerically.
    So we return K_cv = K_gl (copy).

    Parameters
    ----------
    K_gl      : (3,3) intrinsics (fx, fy, cx, cy) in pixel units
    pose_gl   : (3,4) or (4,4) extrinsics in OpenGL convention
                - If pose_type == "c2w": pose maps camera coords to world (C2W)
                - If pose_type == "w2c": pose maps world coords to camera (W2C)
    pose_type : str, "c2w" or "w2c"

    Returns
    -------
    K_cv    : (3,3) intrinsics (unchanged numerically)
    pose_cv : (3,4) or (4,4) extrinsics in OpenCV convention (same shape as input)

    Notes
    -----
    - This assumes row-major matrices with the last column as translation for (3,4)/(4,4).
    - The transform S is its own inverse (S == S^{-1}).
    - If your downstream code expects 3x4, we preserve the input shape.
    """
    # Normalize pose to 4x4
    pose_gl = np.asarray(pose_gl, dtype=np.float32)
    if pose_gl.shape == (3, 4):
        pose4 = np.vstack([pose_gl, np.array([[0, 0, 0, 1]], dtype=np.float32)])
        out_shape = (3, 4)
    elif pose_gl.shape == (4, 4):
        pose4 = pose_gl.copy()
        out_shape = (4, 4)
    else:
        raise ValueError(f"pose_gl must be (3,4) or (4,4), got {pose_gl.shape}")

    # Fixed OpenGL->OpenCV camera-frame conversion
    S = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    if pose_type.lower() == "c2w":
        # X_w = C2W_gl * X_gl  and  X_gl = S * X_cv  =>  C2W_cv = C2W_gl * S
        pose_cv4 = pose4 @ S
    elif pose_type.lower() == "w2c":
        # X_gl = W2C_gl * X_w  and  X_cv = S * X_gl  =>  W2C_cv = S * W2C_gl
        pose_cv4 = S @ pose4
    else:
        raise ValueError("pose_type must be 'c2w' or 'w2c'")

    # Return to original shape
    if out_shape == (3, 4):
        pose_cv = pose_cv4[:3, :]
    else:
        pose_cv = pose_cv4

    K_gl = np.asarray(K_gl, dtype=np.float32)
    if K_gl.shape != (3, 3):
        raise ValueError(f"K_gl must be (3,3), got {K_gl.shape}")

    K_cv = K_gl.copy()  # numerically unchanged
    return K_cv, pose_cv

def get_mesh_eval_points(database):
    if isinstance(database, GlossySyntheticDatabase):
        _, test_ids, _ = get_database_split(database, 'test')
        mesh = trimesh.load_mesh(args.mesh)
        pbar = tqdm(len(test_ids))
        pts_pr = []
        for index, test_id in enumerate(test_ids):
            K = database.get_K(test_id) #(3, 3)
            pose = database.get_pose(test_id) # (3, 4)
            K, pose = gl_to_cv_camera(K, pose, pose_type="c2w")
            print(f"pose shape before inverse: {pose.shape}")
            print(f"K shape: {K.shape}")
            h, w, _ = database.get_image(test_id).shape
            depth_pr, mask_pr = rasterize_depth_map(mesh, pose, K, (h, w)) # (H, W), depth in camera frame
            pts_ = mask_depth_to_pts(mask_pr, depth_pr, K) # (N, 3), in camera frame
            pose = pose_inverse(database.get_pose(test_id)) # (3,4)
            print(f"pose shape: {pose.shape}, pts_ shape: {pts_.shape}, depth_pr shape: {depth_pr.shape}, mask_pr shape: {mask_pr.shape}")
            pts_pr.append(pose_apply(pose, pts_)) # (N, 3), in world frame
            print(f"shape after pose_apply(): {pts_pr[-1].shape}")
            pbar.update(1)

        pts_pr = np.concatenate(pts_pr, 0).astype(np.float32)
        print("Before downsample:", pts_pr.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_pr)
        downpcd = pcd.voxel_down_sample(voxel_size=0.01)
        return np.asarray(downpcd.points,np.float32)
    else:
        raise NotImplementedError

def main():
    database = parse_database_name(f'syn/{args.object}', 'data/GlossySynthetic')
    pts_gt = get_database_eval_points(database)
    pts_pr = get_mesh_eval_points(database)

    dist_gt = nearest_dist(pts_gt, pts_pr, args.batch_size)
    dist_pr = nearest_dist(pts_pr, pts_gt, args.batch_size)

    stem = Path(args.mesh).stem
    chamfer = (np.mean(dist_gt) + np.mean(dist_pr)) / 2
    results = f'{stem} {chamfer:.5f}'
    print(results)
    with open('data/geometry.log','a') as f:
        f.write(results+'\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str,required=True)
    parser.add_argument('--object', type=str,required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    main()