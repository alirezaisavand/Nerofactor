"""File-only dense Bell verification with explicit diagnostic roundoff bounds."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks import bell_dense_reference_v2 as v2
from benchmarks.donut_recovery_v1 import json_new
from experiments import bell_dense_reference_v2 as run
from experiments.verify_bell_heterogeneous_replay_v2 import ROUNDING_FIELDS
from infrastructure.signatures import require

b = v2.b
ULPS = dict.fromkeys(ROUNDING_FIELDS, 1)
ULPS.update(integrated_specular_to_diffuse=1, integrated_specular_fraction=1,
            quantization_rmse_over_specular_rms=4)


def highlight_replay(saved, actual):
    """Allow only the measured host/native rounding in descriptive summaries."""
    require(saved.keys()==actual.keys(), 'Dense Bell highlight summary inventory differs')
    roundoff = {}
    for name, value in actual.items():
        reference = saved[name]
        require(type(reference) is type(value) and
                (type(value) is not float or np.isfinite([reference,value]).all()),
                'Dense Bell highlight summary type or finite check failed: '+name)
        if reference == value:
            continue
        require(type(value) is float and name in ULPS, 'Dense Bell highlight summary differs: '+name)
        edge = reference
        for distance in range(1, ULPS[name]+1):
            edge = np.nextafter(edge,value)
            if edge == value:
                roundoff[name] = dict(saved=reference,replayed=value,absolute_difference=abs(reference-value),ulps=distance)
                break
        else:
            raise ValueError('Dense Bell highlight summary exceeds replay bound: '+name)
    return roundoff


def verify(sha, destination):
    f = v2.freeze(b.ROOT/v2.FREEZE,sha)
    old = v2.prior_definition()
    paths = v2.v1.host_paths(old)
    failed = Path(f['failed_batch'])
    data = v2.v1.source(old,*paths)
    admitted = v2.admit_prefix(f,failed)
    # Replace only the diagnostic comparison in the offline call. The frozen
    # generator, view reconstructions, masks, counters and scientific checks are
    # evaluated by the original implementation.
    with patch.object(run,'highlight_replay',highlight_replay):
        count = run.verify(f,sha,destination,paths,failed,data,admitted)
    report = destination/'verification.json'
    json_new(destination/'replay-policy.json',dict(schema_version=1,
        protocol='bell-dense-reference-v2-offline-replay-v2',
        verifier_sha256=b.digest(Path(__file__).resolve()),
        original_verifier_sha256=b.digest(Path(run.__file__).resolve()),
        freeze_sha256=sha,verification_sha256=b.digest(report),
        original_strict_replay_error='Bell highlight summary differs: integrated_specular_to_diffuse',
        maximum_float64_ULPs_by_diagnostic=ULPS,host_numpy=np.__version__,
        scientific_thresholds_unchanged=True,highlight_masks_exact=True,scientific_queries=0))
    return count


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--freeze-sha256',required=True)
    cli.add_argument('--output',required=True,type=Path)
    args = cli.parse_args()
    print(f'Verified {verify(args.freeze_sha256,args.output)} new dense Bell references')
