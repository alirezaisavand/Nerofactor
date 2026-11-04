

import os
import random
from pathlib import Path

import torch
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.name2dataset import name2dataset
from network.loss import name2loss
from network.renderer import name2renderer
from train.lr_common_manager import name2lr_manager
from network.metrics import name2metrics
from train.train_tools import to_cuda, Logger
from train.train_valid import ValidationEvaluator
from utils.dataset_utils import dummy_collate_fn


class Trainer:
    default_cfg = {
        "optimizer_type": 'adam',
        "multi_gpus": False,
        "lr_type": "exp_decay",
        "lr_cfg": {
            "lr_init": 2.0e-4,
            "lr_step": 100000,
            "lr_rate": 0.5,
        },
        "total_step": 300000,
        "train_log_step": 20,
        "val_interval": 10000,
        "save_interval": 500,
        "novel_view_interval": 10000,
        "worker_num": 8,
        'random_seed': 6033,
    }

    def _init_dataset(self):
        if self.experiment is None:
            self.train_set = name2dataset[self.cfg['train_dataset_type']](self.cfg['train_dataset_cfg'], True)
        else:
            self.train_set = name2dataset[self.cfg['train_dataset_type']](
                self.cfg['train_dataset_cfg'], True, dataset_dir=self.cfg['dataset_dir'],
                preprocessing=self.experiment.preprocessing)
        self.train_set = DataLoader(self.train_set, 1, True, num_workers=self.cfg['worker_num'],
                                    collate_fn=dummy_collate_fn)
        print(f'train set len {len(self.train_set)}')
        self.val_set_list, self.val_set_names = [], []
        dataset_dir = self.cfg['dataset_dir']
        for val_set_cfg in self.cfg['val_set_list']:
            name, val_type, val_cfg = val_set_cfg['name'], val_set_cfg['type'], val_set_cfg['cfg']
            if self.experiment is None:
                val_set = name2dataset[val_type](val_cfg, False, dataset_dir=dataset_dir)
            else:
                val_set = name2dataset[val_type](val_cfg, False, dataset_dir=dataset_dir,
                                               preprocessing=self.experiment.preprocessing)
            val_set = DataLoader(val_set, 1, False, num_workers=self.cfg['worker_num'], collate_fn=dummy_collate_fn)
            self.val_set_list.append(val_set)
            self.val_set_names.append(name)
            print(f'{name} val set len {len(val_set)}')

    def _init_network(self):
        if self.experiment is None:
            self.network = name2renderer[self.cfg['network']](self.cfg).cuda()
        elif self.cfg['network'] == 'shape':
            self.network = name2renderer[self.cfg['network']](
                self.cfg, preprocessing=self.experiment.preprocessing, isolated_geometry=True).cuda()
            self.experiment.record_owner('renderer', self.network.cfg)
        else:
            self.network = name2renderer[self.cfg['network']](
                self.cfg, preprocessing=self.experiment.preprocessing, output_paths=self.experiment.run).cuda()
            self.experiment.record_owner('renderer', self.network.cfg)
            self.experiment.record_owner('shader', self.network.shader_network.cfg)

        # loss
        self.val_losses = []
        for loss_name in self.cfg['loss']:
            self.val_losses.append(name2loss[loss_name](self.cfg))
        self.val_metrics = []

        # metrics
        for metric_name in self.cfg['val_metric']:
            if metric_name in name2metrics:
                if self.experiment is not None and metric_name in ('shape_render', 'mat_render'):
                    self.val_metrics.append(name2metrics[metric_name](self.cfg, output_paths=self.experiment.run))
                else:
                    self.val_metrics.append(name2metrics[metric_name](self.cfg))
            else:
                self.val_metrics.append(name2loss[metric_name](self.cfg))

        # we do not support multi gpu training for NeuRay
        if self.cfg['multi_gpus']:
            raise NotImplementedError
            # make multi gpu network
            # self.train_network=DataParallel(MultiGPUWrapper(self.network,self.val_losses))
            # self.train_losses=[DummyLoss(self.val_losses)]
        else:
            self.train_network = self.network
            self.train_losses = self.val_losses

        if self.cfg['optimizer_type'] == 'adam':
            self.optimizer = Adam
        elif self.cfg['optimizer_type'] == 'sgd':
            self.optimizer = SGD
        else:
            raise NotImplementedError

        self.val_evaluator = ValidationEvaluator(self.cfg)
        self.lr_manager = name2lr_manager[self.cfg['lr_type']](self.cfg['lr_cfg'])
        self.optimizer = self.lr_manager.construct_optimizer(self.optimizer, self.network)

    def __init__(self, cfg, *, experiment=None):
        if experiment is not None and cfg.get('network') not in ('shape', 'material'):
            raise ValueError('Experiment integration requires shape or material')
        self.experiment = experiment
        self.cfg = {**self.default_cfg, **cfg}
        torch.manual_seed(self.cfg['random_seed'])
        np.random.seed(self.cfg['random_seed'])
        random.seed(self.cfg['random_seed'])
        self.model_name = cfg['name']
        self.model_dir = os.path.join('data/model', cfg['name']) if experiment is None else str(experiment.run.checkpoints)
        if not os.path.exists(self.model_dir): Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        self.pth_fn = os.path.join(self.model_dir, 'model.pth')
        self.best_pth_fn = os.path.join(self.model_dir, 'model_best.pth')
        if experiment is not None:
            experiment.record_owner('trainer', self.cfg)

    def run(self):
        self._init_dataset()
        self._init_network()
        self._init_logger()

        best_para, start_step = self._load_model()
        train_iter = iter(self.train_set)

        pbar = tqdm(total=self.cfg['total_step'], bar_format='{r_bar}')
        pbar.update(start_step)

        for step in range(start_step, self.cfg['total_step']):
            try:
                train_data = next(train_iter)
            except StopIteration:
                self.train_set.dataset.reset()
                train_iter = iter(self.train_set)
                train_data = next(train_iter)
            if not self.cfg['multi_gpus']:
                train_data = to_cuda(train_data)
            train_data['step'] = step

            self.train_network.train()
            self.network.train()
            lr = self.lr_manager(self.optimizer, step)

            self.optimizer.zero_grad()
            self.train_network.zero_grad()

            # if (step + 1) % self.cfg['novel_view_interval'] == 0:
            #     render_data = train_data.copy()
            #     render_data["render"] = True

            #     self.train_network(render_data)

            log_info = {}
            outputs = self.train_network(train_data)
            for loss in self.train_losses:
                loss_results = loss(outputs, train_data, step)
                for k, v in loss_results.items():
                    log_info[k] = v

            loss = 0
            loss_str = ""
            for k, v in log_info.items():
                if k.startswith('loss'):
                    loss = loss + torch.mean(v)
                    x = torch.mean(v).detach().cpu().numpy()
                    loss_str += ', ' + k + ": " + f"{x:.4f}"

            loss.backward()
            max_grad_norm = 0.7
            total_norm = torch.nn.utils.clip_grad_norm_(self.train_network.parameters(), max_norm=max_grad_norm)
            self.optimizer.step()
            if ((step + 1) % self.cfg['train_log_step']) == 0:
                self._log_data(log_info, step + 1, 'train')

            if (step + 1) % self.cfg['val_interval'] == 0 or (step + 1) == self.cfg['total_step']:
                torch.cuda.empty_cache()
                val_results = {}
                val_para = 0

                for vi, val_set in enumerate(self.val_set_list):
                    val_results_cur, val_para_cur = self.val_evaluator(
                        self.network, self.val_losses + self.val_metrics, val_set, step,
                        self.model_name, val_set_name=self.val_set_names[vi])
                    for k, v in val_results_cur.items():
                        val_results[f'{self.val_set_names[vi]}-{k}'] = v
                    # always use the final val set to select model!
                    val_para = val_para_cur

                if val_para > best_para:
                    print(f'New best model {self.cfg["key_metric_name"]}: {val_para:.5f} previous {best_para:.5f}')
                    best_para = val_para
                    self._save_model(step + 1, best_para, self.best_pth_fn)
                self._log_data(val_results, step + 1, 'val')
                del val_results, val_para, val_para_cur, val_results_cur

            if (step + 1) % self.cfg['save_interval'] == 0:
                save_fn = None
                self._save_model(step + 1, best_para, save_fn=save_fn)


            pbar.set_postfix(loss=float(loss.detach().cpu().numpy()), lr=lr, others=loss_str)
            pbar.update(1)
            del loss, log_info
        if self.experiment is None:
            logs_dir = os.path.join('logs', self.cfg['name'])
            logs_path = os.path.join(logs_dir, 'logs.txt')
            nvs_imgs_dir = os.path.join('nvs_imgs', self.cfg['name'])
            os.makedirs(logs_dir, exist_ok=True)
            os.makedirs(nvs_imgs_dir, exist_ok=True)
            self.train_network.nvs(log_dir_str=logs_dir,
                                                  imgs_dir=nvs_imgs_dir)
        elif self.cfg['network'] == 'material':
            logs_dir = self.experiment.run.output('logs/nvs')
            nvs_imgs_dir = self.experiment.run.output('renders/nvs')
            logs_dir.mkdir(parents=True, exist_ok=True)
            nvs_imgs_dir.mkdir(parents=True, exist_ok=True)
            self.train_network.nvs(log_dir_str=str(logs_dir), imgs_dir=str(nvs_imgs_dir))
        else:
            print('Final geometry NVS deferred to a separately versioned evaluation protocol; see geometry.json.')

        pbar.close()

    def _load_model(self):
        if self.experiment is not None:
            return self.experiment.restore(self.network, self.optimizer, torch.load, self.cfg['total_step'])
        best_para, start_step = 0, 0
        if os.path.exists(self.pth_fn):
            checkpoint = torch.load(self.pth_fn)
            best_para = checkpoint['best_para']
            start_step = checkpoint['step']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'==> resuming from step {start_step} best para {best_para}')

        return best_para, start_step

    def _save_model(self, step, best_para, save_fn=None):
        save_fn = self.pth_fn if save_fn is None else save_fn
        if self.experiment is not None:
            return self.experiment.save_checkpoint({
                'step': step,
                'best_para': best_para,
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_fn, torch.save)
        torch.save({
            'step': step,
            'best_para': best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_fn)

    def _init_logger(self):
        self.logger = Logger(self.model_dir if self.experiment is None else str(self.experiment.run.logs))

    def _log_data(self, results, step, prefix='train', verbose=False):
        log_results = {}
        for k, v in results.items():
            if isinstance(v, float) or np.isscalar(v):
                log_results[k] = v
            elif type(v) == np.ndarray:
                log_results[k] = np.mean(v)
            else:
                log_results[k] = np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results, prefix, step, verbose)
