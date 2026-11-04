"""Material execution reuses the approved cache/temp/log environment boundary."""

from copy import deepcopy
import traceback

from infrastructure.geometry_execution import execution_environment


def execute_material(experiment, trainer_factory=None):
    experiment.check_output_storage()
    with execution_environment(experiment):
        trainer = None
        try:
            experiment.verify_inputs()
            if trainer_factory is None:
                from train.trainer import Trainer
                trainer_factory = Trainer
            trainer = trainer_factory(deepcopy(experiment.plan.pipeline_config), experiment=experiment)
            trainer.run()
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            writer = getattr(getattr(trainer, "logger", None), "writer", None)
            if writer is not None:
                writer.close()
