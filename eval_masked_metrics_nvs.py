# eval_masked_metrics_nvs.py
import argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from skimage.metrics import structural_similarity as ssim
import json, math, random
from dataclasses import dataclass

# ---- paste/import your GlossyRealDatabase class here ----
# from your_module import GlossyRealDatabase
from dataset.database import GlossyRealDatabase
# ---------------------- split ----------------------
def get_database_split(database, split_type='validation', seed=6033):
    """
    Exactly your function, with seed configurable.
    Returns train_ids, test_ids, nvs_ids
    """
    random.seed(seed)
    img_ids = database.get_img_ids().copy()
    num_nvs_imgs = 5
    random.shuffle(img_ids)
    nvs_ids = img_ids[:num_nvs_imgs]
    if split_type == 'validation':
        test_ids = img_ids[num_nvs_imgs:num_nvs_imgs+1]
        train_ids = img_ids[num_nvs_imgs+1:]
    elif split_type == 'test':
        # If you rely on this path, adapt to your environment
        from utils.io_utils import read_pickle  # or your own helper
        test_ids, train_ids = read_pickle('configs/synthetic_split_128.pkl')
    else:
        raise NotImplementedError
    return train_ids, test_ids, nvs_ids

# ---------------------- helpers ----------------------
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
        scores = [ssim(gt_c[..., c], pr_c[..., c], data_range=1.0) for c in range(3)]
        score = float(np.mean(scores))
    return float(score)

def to_open3d_intrinsics(K, width, height):
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width, height, float(K[0,0]), float(K[1,1]),
                        float(K[0,2]), float(K[1,2]))
    return intr

def to_extrinsic_4x4(R, t):
    ext = np.eye(4, dtype=np.float32)
    ext[:3,:3] = R.astype(np.float32)
    ext[:3, 3] = t.astype(np.float32)
    return ext

def rectified_mesh(mesh, scale, offset, R_rec):
    v = np.asarray(mesh.vertices, dtype=np.float32)
    v_rect = (scale * (v + offset[None, :])) @ R_rec.T
    mesh_out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v_rect),
                                         mesh.triangles)
    mesh_out.triangles = mesh.triangles
    mesh_out.compute_vertex_normals()
    return mesh_out

@dataclass
class EvalItem:
    img_id: int
    img_name: str
    psnr: float
    ssim: float
    saved_masked_render_path: str

# ------------------ mask renderer ------------------
class SilhouetteRenderer:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self.scene = self.renderer.scene
        self.scene.set_background([0, 0, 0, 0])
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.base_color = (1.0, 1.0, 1.0, 1.0)
        self.mesh_id = None

    def set_mesh(self, mesh):
        if self.mesh_id is not None:
            self.scene.remove_geometry(self.mesh_id)
        self.mesh_id = "mesh"
        self.scene.add_geometry(self.mesh_id, mesh, self.mat)

    def set_camera(self, K, extrinsic, width, height):
        intr = to_open3d_intrinsics(K, width, height)
        self.renderer.setup_camera(intr, extrinsic)

    def render_depth_mask(self):
        depth = np.asarray(self.renderer.render_to_depth_image(), dtype=np.float32)
        return np.isfinite(depth) & (depth > 0) & (depth < 1e9)

# ----------------------- main -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--object", type=str, required=True, choices=["bear","bunny","coral","maneki","vase"])
    parser.add_argument("--max_len", type=str, default="raw_1024")
    parser.add_argument("--mesh", type=str, required=True)
    parser.add_argument("--rendered_dir", type=str, required=True)
    parser.add_argument("--split_type", type=str, default="validation", choices=["validation","test"])
    parser.add_argument("--seed", type=int, default=6033)
    parser.add_argument("--save_dir", type=str, default="eval_masked_outputs")
    args = parser.parse_args()

    database_name = f"GlossyReal/{args.object}/{args.max_len}"
    db = GlossyRealDatabase(database_name, args.dataset_dir)

    # --- use your split to fetch exactly 5 nvs ids ---
    _, _, nvs_ids = get_database_split(db, split_type=args.split_type, seed=args.seed)
    if len(nvs_ids) != 5:
        print(f"[warn] nvs_ids size is {len(nvs_ids)}; proceeding with whatever returned.")

    save_dir = Path(args.save_dir)
    masked_render_dir = save_dir / "masked_rendered"
    masked_render_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if not mesh.has_triangles():
        raise RuntimeError("Loaded mesh has no triangles.")

    # Rectify mesh into database coordinates (same as cameras/images)
    mesh_rec = rectified_mesh(mesh, db.scale_rect, db.offset_rect, db.R_rect)

    results = []
    for img_id in nvs_ids:
        K = db.get_K(img_id)
        pose = db.get_pose(img_id)  # [R|t]
        R, t = pose[:, :3], pose[:, 3]
        img_name = db.image_names[img_id]

        gt_path = Path(db.root) / f"images_{db.max_len}" / img_name
        rd_name = img_name.split('.')[0] + ".png"
        rd_path = Path(args.rendered_dir) / rd_name

        gt = imread_rgb(gt_path)
        rd = imread_rgb(rd_path)
        if gt.shape[:2] != rd.shape[:2]:
            rd = cv2.resize(rd, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

        H, W = gt.shape[:2]
        renderer = SilhouetteRenderer(W, H)
        renderer.set_mesh(mesh_rec)
        extrinsic = to_extrinsic_4x4(R, t)
        renderer.set_camera(K, extrinsic, W, H)
        mask = renderer.render_depth_mask()

        gt_f = gt.astype(np.float32) / 255.0
        rd_f = rd.astype(np.float32) / 255.0
        masked_gt = (gt_f * mask[..., None])
        masked_rd = (rd_f * mask[..., None])

        out_path = masked_render_dir / img_name
        imwrite_rgb(out_path, np.clip(masked_rd * 255.0, 0, 255).astype(np.uint8))

        ps = psnr_masked(gt_f, rd_f, mask)
        ss = ssim_masked(gt_f, rd_f, mask)
        results.append(EvalItem(img_id=img_id, img_name=img_name,
                                psnr=ps, ssim=ss,
                                saved_masked_render_path=str(out_path)))
        print(f"[{img_name}]  PSNR(masked): {ps:.3f}   SSIM(masked): {ss:.4f}")

    psnrs = [r.psnr for r in results if np.isfinite(r.psnr)]
    ssims = [r.ssim for r in results if np.isfinite(r.ssim)]
    avg_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
    avg_ssim = float(np.mean(ssims)) if ssims else float("nan")

    metrics = {
        "object": args.object,
        "split_type": args.split_type,
        "seed": args.seed,
        "nvs_ids": nvs_ids,
        "per_image": [
            {"img_id": r.img_id, "img_name": r.img_name,
             "psnr_masked": r.psnr, "ssim_masked": r.ssim,
             "masked_render_path": r.saved_masked_render_path}
            for r in results
        ],
        "average": {"psnr_masked": avg_psnr, "ssim_masked": avg_ssim}
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Summary (NVS) ===")
    for r in results:
        print(f"{r.img_name:30s}  PSNR {r.psnr:.3f}  SSIM {r.ssim:.4f}")
    print(f"Average over {len(results)} (NVS):  PSNR {avg_psnr:.3f}  SSIM {avg_ssim:.4f}")
    print(f"Saved masked rendered images to: {masked_render_dir}")
    print(f"Saved metrics to: {save_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
