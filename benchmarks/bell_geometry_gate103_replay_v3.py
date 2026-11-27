"""Offline-only normal-summary replay rule; scientific gates remain exact."""
import math
import numpy as np

from infrastructure.signatures import require

POLICY='normal-angle-summary-float64-adjacent-v1'
ROUNDING_FIELDS={('signed_normal_error_degrees',stat) for stat in ('median','p95','mean','maximum')}


def compare_surface(recorded,replayed):
    """Reject schema/decision differences; allow one adjacent double in angle summaries.

    The saved normal arrays and all eligibility masks are compared separately and
    exactly. No native threshold or captured result is changed by this comparator.
    """
    accepted=[]
    def compare(a,b,path):
        label='.'.join(path) or '<root>'
        require(type(a) is type(b),'Replay type differs: '+label)
        if isinstance(a,dict):
            require(set(a)==set(b),'Replay keys differ: '+label)
            for key in sorted(a):compare(a[key],b[key],path+(key,))
        elif isinstance(a,float):
            require(math.isfinite(a) and math.isfinite(b),'Nonfinite replay value: '+label)
            if a==b:return
            require(path in ROUNDING_FIELDS and float(np.nextafter(np.float64(a),np.float64(b)))==b,
                    'Replay value differs beyond declared roundoff: '+label)
            accepted.append(dict(field=label,captured=a,replayed=b,absolute_difference=abs(a-b),float64_ULPs=1))
        else:
            require(a==b,'Replay value differs: '+label)
    compare(recorded,replayed,())
    return accepted
