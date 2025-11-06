import argparse, json, math, random
from pathlib import Path
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

import open3d as o3d  # only for reading the PLY and basic ops

import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer, RasterizationSettings, SfMPerspectiveCameras
)
from dataset.database import GlossyRealDatabase

# ----------------- paste or import your GlossyRealDatabase here -----------------
# from your_module import GlossyRealDatabase
# (Use the exact class you posted; it provides poses/Ks/image_names/normalization.)

# ----------------- split (exactly your logic, with configurable seed) ----------
def get_database_split(database, split_type='validation', seed=6033):
    random.seed(seed)
    img_ids = database.get_img_ids().copy()
    num_nvs_imgs = 5
    random.shuffle(img_ids)
    nvs_ids = img_ids[:num_nvs_imgs]
    if split_type == 'validation':
        test_ids = img_ids[num_nvs_imgs:num_nvs_imgs+1]
        train_ids = img_ids[num_nvs_imgs+1:]
    elif split_type == 'test':
        from utils.io_utils import read_pickle  # adjust to your project
        test_ids, train_ids = read_pickle('configs/synthetic_split_128.pkl')
    else:
        raise NotImplementedError
    return train_ids, test_ids, nvs_ids

# ----------------- IO + metrics helpers ---------------------------------------
def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imwrite_rgb(path, img_rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def psnr_masked(gt, pred, mask, data_range=1.0):
    m = mask.astype(np.float32)
    if m.sum() == 0:
        return float("nan")
    diff2 = (gt - pred) ** 2
    mse = (diff2 * m[..., None]).sum() / (m.sum() * gt.shape[2])
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)

def ssim_masked(gt, pred, mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return float("nan")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    gt_c = gt[y0:y1, x0:x1]
    pr_c = pred[y0:y1, x0:x1]
    try:
        score = ssim(gt_c, pr_c, channel_axis=2, data_range=1.0)
    except TypeError:
        score = float(np.mean([ssim(gt_c[..., c], pr_c[..., c], data_range=1.0) for c in range(3)]))
    return float(score)

# ----------------- rectification (match your DB normalization) -----------------
def rectified_mesh_o3d(mesh, scale, offset, R_rec):
    v = np.asarray(mesh.vertices, dtype=np.float32)
    v_rect = (scale * (v + offset[None, :])) @ R_rec.T
    mesh_out = o3d.geometry.TriangleMesh()
    mesh_out.vertices = o3d.utility.Vector3dVector(v_rect.astype(np.float64))
    mesh_out.triangles = mesh.triangles
    return mesh_out

# ----------------- PyTorch3D rasterizer for silhouette masks -------------------
class P3DMaskRenderer:
    def __init__(self, width, height, device="cpu"):
        self.W = int(width)
        self.H = int(height)
        self.device = torch.device(device)
        self.verts = None
        self.faces = None
        self.cameras = None
        self.raster_settings = RasterizationSettings(
            image_size=(self.H, self.W),  # note (H,W)
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
        )

    def set_mesh(self, mesh_o3d):
        v = np.asarray(mesh_o3d.vertices, dtype=np.float32)
        f = np.asarray(mesh_o3d.triangles, dtype=np.int64)
        self.verts = torch.from_numpy(v).to(self.device)
        self.faces = torch.from_numpy(f).to(self.device)

    def set_camera(self, K, extrinsic_4x4):
        R = extrinsic_4x4[:3, :3]
        t = extrinsic_4x4[:3, 3]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        px, py = float(K[0, 2]), float(K[1, 2])

        self.cameras = SfMPerspectiveCameras(
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=self.device),
            principal_point=torch.tensor([[px, py]], dtype=torch.float32, device=self.device),
            R=torch.from_numpy(R).unsqueeze(0).to(self.device).float(),  # world->view
            T=torch.from_numpy(t).unsqueeze(0).to(self.device).float(),  # world->view
            image_size=torch.tensor([[self.H, self.W]], dtype=torch.float32, device=self.device),
            in_ndc=False,
        )

    def render_mask(self):
        mesh = Meshes(verts=[self.verts], faces=[self.faces])
        rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        frags = rasterizer(mesh)  # (1,H,W,K)
        pix_to_face = frags.pix_to_face[0, ..., 0]  # top face index
        mask = (pix_to_face >= 0).cpu().numpy()
        return mask

# ----------------- main --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--object", required=True, choices=["bear","bunny","coral","maneki","vase"])
    ap.add_argument("--max_len", default="raw_1024")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--rendered_dir", required=True)
    ap.add_argument("--split_type", default="validation", choices=["validation","test"])
    ap.add_argument("--seed", type=int, default=6033)
    ap.add_argument("--save_dir", default="eval_masked_outputs")
    ap.add_argument("--device", default="cpu")  # "cpu" or "cuda"
    args = ap.parse_args()

    database_name = f"GlossyReal/{args.object}/{args.max_len}"
    db = GlossyRealDatabase(database_name, args.dataset_dir)

    # get the exact 5 NVS ids
    _, _, nvs_ids = get_database_split(db, split_type=args.split_type, seed=args.seed)

    save_dir = Path(args.save_dir)
    (save_dir / "masked_rendered").mkdir(parents=True, exist_ok=True)

    # load and rectify mesh so it aligns with db poses (same as images)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(mesh.triangles) == 0:
        raise RuntimeError("GT mesh has no triangles.")
    mesh_rec = rectified_mesh_o3d(mesh, db.scale_rect, db.offset_rect, db.R_rect)

    results = []
    for img_id in nvs_ids:
        K = db.get_K(img_id)
        pose = db.get_pose(img_id)  # [R|t], world->cam
        R, t = pose[:, :3], pose[:, 3]
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3]  = t

        img_name = db.image_names[img_id]
        img_name = img_name.replace(".jpg", ".png")  # rendered images are .png
        gt_path = Path(db.root) / f"images_{db.max_len}" / img_name
        rd_path = Path(args.rendered_dir) / img_name  # assumes same filenames

        gt = imread_rgb(gt_path)
        rd = imread_rgb(rd_path)
        if gt.shape[:2] != rd.shape[:2]:
            rd = cv2.resize(rd, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        H, W = gt.shape[:2]

        # rasterize mask with PyTorch3D
        renderer = P3DMaskRenderer(W, H, device=args.device)
        renderer.set_mesh(mesh_rec)
        renderer.set_camera(K, extrinsic)
        mask = renderer.render_mask()

        gt_f = gt.astype(np.float32) / 255.0
        rd_f = rd.astype(np.float32) / 255.0

        # save masked rendered image (black bg)
        masked_rd = (rd_f * mask[..., None])
        out_path = save_dir / "masked_rendered" / img_name
        imwrite_rgb(out_path, (np.clip(masked_rd, 0, 1) * 255).astype(np.uint8))

        # metrics on masked region
        ps = psnr_masked(gt_f, rd_f, mask)
        ss = ssim_masked(gt_f, rd_f, mask)
        results.append({
            "img_id": int(img_id),
            "img_name": img_name,
            "psnr_masked": ps,
            "ssim_masked": ss,
            "masked_render_path": str(out_path)
        })
        print(f"[{img_name}]  PSNR(masked): {ps:.3f}  SSIM(masked): {ss:.4f}")

    # averages
    psnrs = [r["psnr_masked"] for r in results if np.isfinite(r["psnr_masked"])]
    ssims = [r["ssim_masked"] for r in results if np.isfinite(r["ssim_masked"])]
    avg_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
    avg_ssim = float(np.mean(ssims)) if ssims else float("nan")

    out_json = {
        "object": args.object,
        "split_type": args.split_type,
        "seed": args.seed,
        "nvs_ids": list(map(int, nvs_ids)),
        "per_image": results,
        "average": {"psnr_masked": avg_psnr, "ssim_masked": avg_ssim}
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(out_json, f, indent=2)

    print("\n=== Summary (NVS) ===")
    for r in results:
        print(f"{r['img_name']:30s}  PSNR {r['psnr_masked']:.3f}  SSIM {r['ssim_masked']:.4f}")
    print(f"Average over {len(results)}:  PSNR {avg_psnr:.3f}  SSIM {avg_ssim:.4f}")
    print(f"Saved masked renders → {save_dir / 'masked_rendered'}")
    print(f"Saved metrics → {save_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
