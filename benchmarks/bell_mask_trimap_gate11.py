"""Prospective image-only trimap rule; no SAM2 import or segmentation inference."""
import numpy as np
from scipy.ndimage import binary_dilation
from infrastructure.signatures import require

FOREGROUND_LOGIT=1.0
BACKGROUND_LOGIT=-1.0
UNCERTAINTY_MARGIN=4


def trimap(scores,object_logits):
    """Four fixed full-resolution decision-score maps; preserve P1-forward FG."""
    scores=np.asarray(scores);object_logits=np.asarray(object_logits)
    require(scores.ndim==3 and scores.shape[0]==4 and min(scores.shape[1:])>0 and
            scores.dtype.kind=='f' and np.isfinite(scores).all() and
            object_logits.shape==(4,) and np.isfinite(object_logits).all(),'Finite four-pass score inventory required')
    foreground=scores[0]>FOREGROUND_LOGIT
    plausible=np.any(scores>=BACKGROUND_LOGIT,axis=0)
    uncertain_margin=binary_dilation(plausible,structure=np.ones((3,3)),iterations=UNCERTAINTY_MARGIN,border_value=0)
    background=np.all(scores<BACKGROUND_LOGIT,axis=0)&~uncertain_margin&~foreground
    if not (object_logits>0).all():background[:]=False
    uncertain=~(foreground|background)
    require(not (foreground&background).any(),'Trimap labels overlap')
    return dict(foreground=foreground,background=background,uncertain=uncertain)
