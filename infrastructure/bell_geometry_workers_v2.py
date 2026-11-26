"""Spawn-only process transport for pinned G's payload-free dummy loaders."""
from contextlib import contextmanager

from infrastructure.signatures import require
from infrastructure.stage_sources import PINS, baseline, imported_sources


def validate_worker(worker_id):
    import torch
    info=torch.utils.data.get_worker_info()
    require(info is not None and info.id==worker_id and info.num_workers==8,
            'Unexpected Bell worker identity')
    require(not torch.cuda.is_initialized(),'Bell dummy worker initialized CUDA')
    root,_=baseline('geometry',PINS['geometry'])
    imported_sources(root)  # Spawn inherits the selected source path, not the parent's import finder.


@contextmanager
def spawned_loaders(trainer_module, report):
    original=trainer_module.DataLoader
    def build(*args,**kwargs):
        require(kwargs.get('num_workers')==8 and
                'multiprocessing_context' not in kwargs and 'worker_init_fn' not in kwargs,
                'Unexpected original G loader settings')
        loader=original(*args,**kwargs,multiprocessing_context='spawn',worker_init_fn=validate_worker)
        require(loader.multiprocessing_context.get_start_method()=='spawn','Spawn worker context unavailable')
        report.setdefault('loader_transport',[]).append(dict(workers=8,start_method='spawn',
                                                            dataset=type(loader.dataset).__name__))
        return loader
    trainer_module.DataLoader=build
    try:
        yield
    finally:
        trainer_module.DataLoader=original
