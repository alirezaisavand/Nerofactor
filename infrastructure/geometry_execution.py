"""Execution boundary: runtime output routing precedes scientific imports."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import traceback

from infrastructure.geometry import CODE_ROOT


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.log.write(text)
        return self.original.write(text)

    def flush(self):
        self.log.flush()
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def execution_environment(experiment):
    run = experiment.run
    destinations = {
        "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
        "XDG_CACHE_HOME": "cache/runtime/xdg", "XDG_CONFIG_HOME": "cache/runtime/config",
        "XDG_DATA_HOME": "cache/runtime/data", "MPLCONFIGDIR": "cache/runtime/matplotlib",
        "TORCH_HOME": "cache/runtime/torch", "TORCH_EXTENSIONS_DIR": "cache/runtime/torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": "cache/runtime/inductor", "TRITON_CACHE_DIR": "cache/runtime/triton",
        "NUMBA_CACHE_DIR": "cache/runtime/numba", "CUDA_CACHE_PATH": "cache/runtime/cuda",
    }
    values = {}
    for name, relative in destinations.items():
        path = run.output(relative)
        path.mkdir(parents=True, exist_ok=True)
        values[name] = str(path)
    values["PYTHONDONTWRITEBYTECODE"] = "1"
    previous = {name: os.environ.get(name) for name in values}
    previous_temp, previous_bytecode, previous_cwd = tempfile.tempdir, sys.dont_write_bytecode, Path.cwd()
    try:
        os.environ.update(values)
        tempfile.tempdir = values["TMPDIR"]
        sys.dont_write_bytecode = True
        # Preserve the baseline relative *input* paths for BSDF assets and split
        # definitions. Audited output writers use explicit experiment paths.
        os.chdir(CODE_ROOT)
        with run.output("logs/console.log").open("a") as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                yield
    finally:
        os.chdir(previous_cwd)
        tempfile.tempdir, sys.dont_write_bytecode = previous_temp, previous_bytecode
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def execute_geometry(experiment, trainer_factory=None):
    """Only explicit callers enter model execution; tests inject a CPU fake."""
    with execution_environment(experiment):
        trainer = None
        try:
            if trainer_factory is None:
                from train.trainer import Trainer
                trainer_factory = Trainer
            trainer = trainer_factory(deepcopy(experiment.plan.pipeline_config), experiment=experiment)
            trainer.run()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            logger = getattr(trainer, "logger", None)
            writer = getattr(logger, "writer", None)
            if writer is not None:
                writer.close()
