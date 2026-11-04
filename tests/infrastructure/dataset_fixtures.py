"""Actual dataset definitions with explicit stand-ins for missing native backends.

The stand-ins test storage/control-flow parity, not OpenCV/Open3D numerics.
No scientific production imports or original datasets are used.
"""

import abc
import ast
import builtins
from contextlib import contextmanager
import glob
import json
import io
import os
from pathlib import Path
import pickle
import random
import subprocess
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import imageio
import numpy as np
from skimage.io import imread, imsave

from colmap import plyfile
from colmap.read_write_model import Camera, Image, read_model, write_model


REPO = Path(__file__).resolve().parents[2]
BASELINE = "2f40ca311950a6323727508ac6470f3a21beb768"


def baseline_source():
    return subprocess.check_output(["git", "show", BASELINE + ":dataset/database.py"], cwd=REPO, text=True)


def fixture_resize(img, ratio):
    h, w = img.shape[:2]
    rows = np.minimum((np.arange(int(round(h * ratio))) / ratio).astype(int), h - 1)
    cols = np.minimum((np.arange(int(round(w * ratio))) / ratio).astype(int), w - 1)
    return img[rows[:, None], cols]


def fixture_crop(img, K, pose, position, angle, scale, h, w):
    out = fixture_resize(img, h / img.shape[0])[:, :w]
    K1 = K.copy()
    K1[0, 2] += position[0] * 0.01
    return out, K1, pose.copy(), np.eye(3), np.eye(3)


def cv_read(path, flags=None):
    arr = imageio.imread(path)
    if flags == 0:
        return arr if arr.ndim == 2 else arr[..., 0]
    return arr[..., :3][..., ::-1]


class PointCloud:
    def voxel_down_sample(self, **kwargs):
        raise AssertionError("Open3D downsampling is not validated by a stand-in")


def write_points(path, cloud):
    np.savetxt(path, np.asarray(cloud.points))
    return True


def load_module(source):
    module = ModuleType("dataset.database")
    scope = module.__dict__
    scope.update(abc=abc, glob=glob, os=os, random=random, Path=Path, np=np,
                 json=json, pickle=pickle, imageio=imageio, imread=imread, imsave=imsave,
                 tqdm=lambda values, **kwargs: values, plyfile=plyfile, read_model=read_model,
                 resize_img=fixture_resize, look_at_crop=fixture_crop,
                 cv2=SimpleNamespace(imread=cv_read, IMREAD_GRAYSCALE=0, COLOR_BGR2RGB=1,
                                     cvtColor=lambda arr, code: arr[..., ::-1]),
                 o3d=SimpleNamespace(geometry=SimpleNamespace(PointCloud=PointCloud),
                                     utility=SimpleNamespace(Vector3dVector=np.asarray),
                                     io=SimpleNamespace(write_point_cloud=write_points)))
    names = {"read_pickle", "save_pickle", "project_points", "pose_inverse", "pose_apply",
             "transform_points_pose", "pts_to_hpts", "hpts_to_pts", "mask_depth_to_pts"}
    utils = ast.parse((REPO / "utils/base_utils.py").read_text())
    functions = [node for node in utils.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "fixture_original_utils", "exec"), scope)
    tree = ast.parse(source)
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "fixture_dataset", "exec"), scope)
    return module


def create_real(family, scene="bear", existing=False):
    root = family / scene
    (root / "images").mkdir(parents=True)
    (root / "colmap/sparse/0").mkdir(parents=True)
    cameras = {1: Camera(id=1, model="SIMPLE_PINHOLE", width=8, height=8,
                         params=np.array([4.0, 4.0, 4.0]))}
    images = {}
    for i in range(8):
        name = "{:04}.png".format(i + 1)
        arr = (np.arange(8 * 8 * 3).reshape(8, 8, 3) + i).astype(np.uint8)
        imageio.imwrite(root / "images" / name, arr)
        images[i + 1] = Image(id=i + 1, qvec=np.array([1., 0., 0., 0.]), tvec=np.array([0., 0., 4.]),
                              camera_id=1, name=name, xys=np.empty((0, 2)), point3D_ids=np.empty(0, dtype=np.int64))
    write_model(cameras, images, {}, str(root / "colmap/sparse/0"))
    points = np.array([(-1., -1., -1.), (1., -1., -1.), (-1., 1., -1.),
                       (1., 1., 1.), (1., -1., 1.), (-1., 1., 1.)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    plyfile.PlyData([plyfile.PlyElement.describe(points, "vertex")], text=True).write(str(root / "object_point_cloud.ply"))
    if scene != "bear":
        np.savetxt(root / "meta_info.txt", [[0, 1, 0], [0, 0, 1]])
    if existing:
        poses = {i: np.concatenate([np.eye(3), np.array([[0.], [0.], [4.]])], axis=1).astype(np.float32) for i in images}
        Ks = {i: np.array([[4., 0, 4.], [0, 4., 4.], [0, 0, 1]], np.float32) for i in images}
        with (root / "cache.pkl").open("wb") as stream:
            pickle.dump([poses, Ks, {i: images[i].name for i in images}, list(images)], stream)
        (root / "images_raw_4/masks_train").mkdir(parents=True)
        for i in images:
            imageio.imwrite(root / "images_raw_4" / images[i].name, np.full((4, 4, 3), 80 + i, np.uint8))
            imageio.imwrite(root / "images_raw_4/masks_train" / images[i].name, np.full((4, 4), 255, np.uint8))
    return root


def create_nerf(family):
    root = family / "scene"
    for folder in ["train", "masks_train", "depth"]:
        (root / folder).mkdir(parents=True)
    frames = []
    for i in range(8):
        name = "{:04}.png".format(i + 1)
        imageio.imwrite(root / "train" / name, np.full((3, 4, 4), 80 + i, np.uint8))
        imageio.imwrite(root / "masks_train" / name, np.full((3, 4), 255, np.uint8))
        imageio.imwrite(root / "depth" / ("Image" + name), np.full((3, 4), 1000, np.uint16))
        pose = np.eye(4)
        pose[:3, 3] = [0, 0, 2]
        frames.append({"file_path": "train\\" + name, "transform_matrix": pose.tolist()})
    (root / "transforms_train.json").write_text(json.dumps({"camera_angle_x": 0.7, "frames": frames}))
    return root


def create_synthetic(family):
    root = family / "scene"
    (root / "masks_train").mkdir(parents=True)
    for i in range(8):
        with (root / (str(i) + "-camera.pkl")).open("wb") as stream:
            pickle.dump([np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1), np.eye(3)], stream)
        imageio.imwrite(root / (str(i) + ".png"), np.full((3, 4, 3), 90 + i, np.uint8))
        imageio.imwrite(root / (str(i) + "-depth.png"), np.full((3, 4), 1000, np.uint16))
        imageio.imwrite(root / "masks_train" / (str(i) + ".png"), np.full((3, 4), 255, np.uint8))
    return root


def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@contextmanager
def deny_input_writes(root):
    """Detect attempted input writes without chmod or read-only mount probes."""
    root = root.resolve()

    def check(path):
        if isinstance(path, (str, os.PathLike)):
            resolved = Path(path).resolve()
            if resolved == root or root in resolved.parents:
                raise AssertionError("Attempted input write: {}".format(path))

    def guard_open(original):
        def guarded(path, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                check(path)
            return original(path, mode, *args, **kwargs)
        return guarded

    original_mkdir = Path.mkdir

    def mkdir(path, *args, **kwargs):
        check(path)
        return original_mkdir(path, *args, **kwargs)

    with patch("builtins.open", guard_open(builtins.open)), patch("io.open", guard_open(io.open)), patch.object(Path, "mkdir", mkdir):
        yield
