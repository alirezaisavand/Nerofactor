"""Explicit O1 derivative; authoritative G and its original Trainer stay intact."""
from contextlib import contextmanager
from infrastructure.signatures import require

VARIANT='G_background_opacity_oracle_v1'
BACKGROUND_COEFFICIENT=1.0


def renderer_class(parent,observer=None):
    """Add one reduction after the original train_step, retaining its exact L1."""
    class BackgroundOpacityRenderer(parent):
        geometry_variant=VARIANT

        def train_step(self,step):
            require(not self.is_nerf,'O1 is restricted to the frozen Bell acquisition')
            start=self.train_batch_i;rn=self.cfg['train_ray_num']
            # Keep the old storage alive if original train_step replaces the
            # shuffled dictionary before returning. Do not read the new cursor.
            mask=self.train_batch['masks'][start:start+rn]
            outputs=super().train_step(step)
            acc=outputs['fg_acc'];foreground=outputs['loss_mask']
            require(mask.shape==acc.shape and acc.ndim==1 and len(acc)>0,
                    'Original foreground mask and accumulation shapes differ')
            background=((1-mask.to(device=acc.device))*acc).mean()
            outputs['loss_mask']=foreground+BACKGROUND_COEFFICIENT*background
            if observer is not None:observer(step,foreground.detach(),background.detach(),outputs['loss_mask'].detach())
            return outputs

    return BackgroundOpacityRenderer


@contextmanager
def select(renderer,observer=None):
    parent=renderer.NeROShapeRenderer
    require(renderer.name2renderer['shape'] is parent,'Expected original G renderer dispatch')
    derivative=renderer_class(parent,observer)
    renderer.name2renderer['shape']=derivative
    try:yield derivative
    finally:renderer.name2renderer['shape']=parent
