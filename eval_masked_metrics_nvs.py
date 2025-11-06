import argparse, json, math, random
from pathlib import Path
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import open3d as o3d  # only for PLY IO
from dataset.database import GlossyRealDatabase

# ----------------- paste/import your GlossyRealDatabase here -----------------
# from your_module import GlossyRealDatabase

# ----------------- same split logic you gave -----------------
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
        from utils.io_utils import read_pickle  # adjust if you use this branch
        test_ids, train_ids = read_pickle('configs/synthetic_split_128.pkl')
    else:
        raise NotImplementedError
    return train_ids, test_ids, nvs_ids

# ----------------- helpers -----------------
def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imwrite_rgb(path, img_rgb):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
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
        return float(ssim(gt_c, pr_c, channel_axis=2, data_range=1.0))
    except TypeError:
        return float(np.mean([ssim(gt_c[..., c], pr_c[..., c], data_range=1.0) for c in range(3)]))

def rectified_mesh_vertices_faces(mesh, scale, offset, R_rec):
    # x3 = R_rec @ (scale * (x0 + offset))
    v = np.asarray(mesh.vertices, dtype=np.float32)
    v_rect = (scale * (v + offset[None, :])) @ R_rec.T
    f = np.asarray(mesh.triangles, dtype=np.int32)
    return v_rect, f

# ----------------- CPU silhouette rasterization (union of projected triangles) -----------------
def project_points(K, R, t, V):
    """
    V: (N,3) world coords (already rectified).
    R,t: world->cam (OpenCV) so x_c = R x_w + t.
    Returns: uv (N,2), Z (N,), valid (N,)
    """
    Xc = (R @ V.T + t[:, None]).T  # (N,3)
    Z = Xc[:, 2]
    valid = Z > 1e-6  # simple near-plane
    uv = np.empty((V.shape[0], 2), dtype=np.float32); uv[:] = np.nan
    uv[valid, 0] = K[0, 0] * (Xc[valid, 0] / Z[valid]) + K[0, 2]
    uv[valid, 1] = K[1, 1] * (Xc[valid, 1] / Z[valid]) + K[1, 2]
    return uv, Z, valid

def estimate_front_ratio(K, R, t, V, H, W, sample=20000):
    """
    Heuristic to decide if (R,t) is world->cam:
    - fraction of sampled vertices with Z>0
    - fraction projected inside image
    """
    n = V.shape[0]
    idx = np.random.choice(n, size=min(n, sample), replace=False)
    uv, Z, valid = project_points(K, R, t, V[idx])
    front = valid.mean()
    if front <= 0:
        return 0.0, 0.0
    u, v = uv[:, 0], uv[:, 1]
    inside = np.logical_and.reduce((
        valid,
        u >= 0, u < W,
        v >= 0, v < H,
    )).mean()
    return float(front), float(inside)

def invert_pose(R, t):
    """Convert camera->world to world->camera."""
    Rinv = R.T
    tinv = -Rinv @ t
    return Rinv, tinv

def resolve_pose_direction(K, R_in, t_in, V, H, W):
    """
    Try both interpretations:
    - assume (R_in, t_in) is world->cam (w2c)
    - assume it's camera->world (c2w), so invert to w2c
    Pick the one that yields more in-front & on-image verts.
    """
    front_a, inside_a = estimate_front_ratio(K, R_in, t_in, V, H, W)
    Rb, tb = invert_pose(R_in, t_in)
    front_b, inside_b = estimate_front_ratio(K, Rb, tb, V, H, W)

    score_a = front_a * 0.7 + inside_a * 0.3
    score_b = front_b * 0.7 + inside_b * 0.3

    if score_b > score_a:
        return Rb.astype(np.float32), tb.astype(np.float32), {"used": "inverted(c2w->w2c)", "front": front_b, "inside": inside_b}
    else:
        return R_in.astype(np.float32), t_in.astype(np.float32), {"used": "as-is(w2c)", "front": front_a, "inside": inside_a}

def build_silhouette_mask(V, F, K, R_in, t_in, H, W, debug_print=False):
    """
    Union-of-triangles silhouette using OpenCV fill. Auto-resolves pose direction.
    """
    # pick the better pose direction
    R, t, dbg = resolve_pose_direction(K, R_in, t_in, V, H, W)
    if debug_print:
        print(f"pose pick: {dbg['used']}, front={dbg['front']:.3f}, inside={dbg['inside']:.3f}")

    mask = np.zeros((H, W), dtype=np.uint8)
    uv, Z, valid = project_points(K, R, t, V)

    # Fill each triangle
    for tri in F:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        pts = np.array([uv[i0], uv[i1], uv[i2]], dtype=np.float32)

        minx, miny = np.floor(pts[:,0].min()), np.floor(pts[:,1].min())
        maxx, maxy = np.ceil(pts[:,0].max()),  np.ceil(pts[:,1].max())
        if maxx < 0 or maxy < 0 or minx >= W or miny >= H:
            continue

        cv2.fillConvexPoly(mask, pts.astype(np.int32), 255, lineType=cv2.LINE_AA)

    return mask.astype(bool)

# ----------------- main -----------------
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
    args = ap.parse_args()

    database_name = f"GlossyReal/{args.object}/{args.max_len}"
    db = GlossyRealDatabase(database_name, args.dataset_dir)

    # exact 5 NVS ids
    _, _, nvs_ids = get_database_split(db, split_type=args.split_type, seed=args.seed)

    save_dir = Path(args.save_dir)
    (save_dir / "masked_rendered").mkdir(parents=True, exist_ok=True)

    # load mesh & rectify into DB coords
    gt_mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(gt_mesh.triangles) == 0:
        raise RuntimeError("GT mesh has no triangles.")
    V_rect, F = rectified_mesh_vertices_faces(gt_mesh, db.scale_rect, db.offset_rect, db.R_rect)

    results = []
    for img_id in nvs_ids:
        K = db.get_K(img_id)
        pose = db.get_pose(img_id)  # [R|t], world->cam
        R, t = pose[:, :3].astype(np.float32), pose[:, 3].astype(np.float32)

        img_name = db.image_names[img_id]
        rd_img_name = img_name.split('.')[0] + ".png"  # ensure .png
        gt_path = Path(db.root) / f"images_{db.max_len}" / img_name
        rd_path = Path(args.rendered_dir) / rd_img_name

        gt = imread_rgb(gt_path)
        rd = imread_rgb(rd_path)
        if gt.shape[:2] != rd.shape[:2]:
            rd = cv2.resize(rd, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        H, W = gt.shape[:2]

        # build silhouette mask by projecting all triangles
        mask = build_silhouette_mask(V_rect, F, K, R, t, H, W)

        gt_f = gt.astype(np.float32) / 255.0
        rd_f = rd.astype(np.float32) / 255.0

        # save masked rendered image
        masked_rd = (rd_f * mask[..., None])
        out_path = save_dir / "masked_rendered" / img_name
        imwrite_rgb(out_path, (np.clip(masked_rd, 0, 1) * 255).astype(np.uint8))

        # metrics over masked region (tight bbox)
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
