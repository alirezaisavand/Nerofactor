"""CPU plumbing fakes around unchanged trainer/evaluator/metric definitions."""

import ast
from collections import OrderedDict
from contextlib import nullcontext
import os
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np
from skimage.io import imsave
from skimage.metrics import structural_similarity


REPO = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
        self.shape, self.dtype = self.value.shape, self.value.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def backward(self):
        pass

    def __add__(self, other):
        return Tensor(self.value + (other.value if isinstance(other, Tensor) else other))

    __radd__ = __add__

    def __mul__(self, other):
        return Tensor(self.value * other)


def compile_nodes(path, scope, predicate):
    nodes = [n for n in ast.parse((REPO / path).read_text()).body if predicate(n)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), scope)


def defaults(path, cls):
    node = next(n for n in ast.parse((REPO / path).read_text()).body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(ast.literal_eval(n.value) for n in node.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "default_cfg" for t in n.targets))


def trainer_fixture(*, legacy_nvs=False):
    events, parses, networks, metrics, writers = [], [], [], [], []

    def parse_database(name, root, *, preprocessing=None):
        parses.append((name, root, preprocessing))
        if preprocessing is not None:
            preprocessing.bind(name, Path(root) / name.split("/")[1], {"testskip": 8})
        return SimpleNamespace(get_img_ids=lambda: [str(i) for i in range(8)])

    class Optimizer:
        def __init__(self, parameters, **kwargs):
            self.param_groups = [{**kwargs, "params": [0]}]
            self.loaded = None

        def zero_grad(self):
            pass

        def step(self):
            events.append("optimizer_step")

        def state_dict(self):
            return {"param_groups": self.param_groups, "state": {}}

        def load_state_dict(self, value):
            self.loaded = value

    class Writer:
        def __init__(self, log_dir):
            self.root = Path(log_dir)
            self.root.mkdir(exist_ok=True, parents=True)
            (self.root / "events.out.tfevents.fixture").write_bytes(b"CPU writer fixture")
            self.closed = False
            writers.append(self)

        def add_scalar(self, *args):
            pass

        def close(self):
            self.closed = True

    def save(state, path):
        events.append(("save", str(path)))
        with open(path, "wb") as stream:
            pickle.dump(state, stream)

    def load(path):
        events.append(("load", str(path)))
        with open(path, "rb") as stream:
            return pickle.load(stream)

    torch = SimpleNamespace(Tensor=Tensor, manual_seed=lambda seed: events.append(("torch_seed", seed)),
                            mean=lambda tensor: Tensor(np.mean(tensor.value)), no_grad=nullcontext,
                            cuda=SimpleNamespace(empty_cache=lambda: None), save=save, load=load,
                            nn=SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=lambda pars, max_norm: events.append(("clip", max_norm)))))
    loss_scope = {"np": np, "torch": torch}
    compile_nodes("network/loss.py", loss_scope, lambda n: isinstance(n, ast.ClassDef))
    compile_nodes("network/loss.py", loss_scope, lambda n: isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "name2loss" for t in n.targets))
    split_scope = {"random": random, "BaseDatabase": object}
    compile_nodes("dataset/database.py", split_scope, lambda n: isinstance(n, ast.FunctionDef) and n.name == "get_database_split")
    dataset_scope = {"Dataset": object, "parse_database_name": parse_database,
                     "get_database_split": split_scope["get_database_split"]}
    compile_nodes("dataset/train_dataset.py", dataset_scope, lambda n: isinstance(n, ast.ClassDef))
    metric_scope = {"Path": Path, "np": np, "imsave": imsave, "Loss": loss_scope["Loss"],
                    "structural_similarity": structural_similarity,
                    "color_map_backward": lambda values: np.clip(values * 255, 0, 255).astype(np.uint8),
                    "draw_materials": lambda *args: [],
                    "concat_images_list": lambda *values, vert=False: np.concatenate(values, axis=0 if vert else 1)}
    compile_nodes("network/metrics.py", metric_scope, lambda n: isinstance(n, ast.FunctionDef) and n.name in {"compute_psnr", "psnr"})
    compile_nodes("network/metrics.py", metric_scope, lambda n: isinstance(n, ast.ClassDef) and n.name == "ShapeRenderMetrics")

    def metric(cfg, **kwargs):
        result = metric_scope["ShapeRenderMetrics"](cfg, **kwargs)
        metrics.append(result)
        return result

    class Renderer:
        def __init__(self, cfg, *, preprocessing=None, isolated_geometry=False):
            self.cfg = {**defaults("network/renderer.py", "NeROShapeRenderer"), **cfg}
            self.isolated_geometry = isolated_geometry
            self.loaded = None
            parse_database(cfg["database_name"], cfg["dataset_dir"], preprocessing=preprocessing)
            # Actual split initialization is exercised separately with device fakes.
            # Legacy tail characterization alone uses a material-style stand-in.
            if legacy_nvs:
                self.nvs = self.legacy_nvs
            networks.append(self)

        def cuda(self):
            return self

        def parameters(self):
            return [0]

        def state_dict(self):
            return {"geometry": Tensor([1])}

        def load_state_dict(self, state):
            self.loaded = state

        def train(self):
            pass

        def eval(self):
            pass

        def zero_grad(self):
            pass

        def __call__(self, data):
            return {"loss_rgb": Tensor([0.2]), "gt_rgb": Tensor(np.full((12, 12, 3), 0.25)),
                    "ray_rgb": Tensor(np.full((12, 12, 3), 0.5)), "normal": Tensor(np.zeros((12, 12, 3)))}

        def nvs(self, pose, K, h, w):
            raise AssertionError("Isolated geometry must defer final NVS")

        def legacy_nvs(self, *, log_dir_str, imgs_dir):
            events.append(("nvs", log_dir_str, imgs_dir))
            Path(log_dir_str, "fixture.txt").write_text("fake only; actual shape NVS has a different signature")

    class Loader:
        def __init__(self, dataset, *args, **kwargs):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __iter__(self):
            return iter([{}] if self.dataset.is_train else [{"index": 0}])

    class Progress:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, *args):
            pass

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass

    logger_scope = {"os": os, "OrderedDict": OrderedDict, "SummaryWriter": Writer}
    compile_nodes("train/train_tools.py", logger_scope, lambda n: isinstance(n, ast.ClassDef) and n.name == "Logger")
    evaluator_scope = {"np": np, "torch": torch, "time": SimpleNamespace(time=lambda: 0),
                       "tqdm": lambda value: value, "to_cuda": lambda value: value,
                       "name2key_metrics": {"psnr": metric_scope["psnr"]}}
    compile_nodes("train/train_valid.py", evaluator_scope, lambda n: isinstance(n, ast.ClassDef))
    import abc
    lr_scope = {"abc": abc, "np": np}
    compile_nodes("train/lr_common_manager.py", lr_scope, lambda n: isinstance(n, ast.ClassDef))
    scope = {"os": os, "random": random, "Path": Path, "torch": torch, "np": np,
             "Adam": Optimizer, "SGD": Optimizer, "DataLoader": Loader, "tqdm": Progress,
             "name2dataset": {"dummy": dataset_scope["DummyDataset"]}, "name2renderer": {"shape": Renderer},
             "name2loss": loss_scope["name2loss"], "name2metrics": {"shape_render": metric},
             "name2lr_manager": {"warm_up_cos": lr_scope["WarmUpCosLR"]},
             "to_cuda": lambda value: value, "Logger": logger_scope["Logger"],
             "ValidationEvaluator": evaluator_scope["ValidationEvaluator"], "dummy_collate_fn": lambda x: x}
    compile_nodes("train/trainer.py", scope, lambda n: isinstance(n, ast.ClassDef))
    return SimpleNamespace(Trainer=scope["Trainer"], events=events, parses=parses, networks=networks,
                           metrics=metrics, writers=writers, torch=torch, Optimizer=Optimizer,
                           Renderer=Renderer, metric_scope=metric_scope)
