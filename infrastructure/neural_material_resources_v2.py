"""Saved-evidence storage and memory accounting; no model or renderer calls."""
import json
import math
import os
from pathlib import Path
import shutil

from infrastructure import append_only_scientific_log_v1 as log
from infrastructure.signatures import require

ROOT=Path(__file__).resolve().parents[1]
D2='docs/evidence/NEURAL_MATERIAL_REPORTING_INTEGRATION_V2.json'
D2_SHA='5d4f9058b93e119236abbc967d75b9c5821d95409458eeb96f0ab281c10a5700'


def selected(path,sha):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and path.is_file() and log.file_sha(path)==sha,'Selected evidence identity mismatch: '+str(path))
    return log.decode(path.read_bytes())


def tensor_bytes(catalog):
    sizes={'torch.float32':4,'torch.float64':8,'torch.int64':8,'torch.uint8':1}
    return sum(math.prod(m['shape'])*sizes[m['dtype']] for m in catalog.values())


def budget(original):
    evidence=selected(ROOT/D2,D2_SHA)
    full=evidence['reporting_benchmarks']['complete'];censor=evidence['reporting_benchmarks']['censored']
    for record in (full['measurement'],censor['measurement'],evidence['field_only']['measurement']):selected(record['path'],record['sha256'])
    field=selected(evidence['field_only']['measurement']['path'],evidence['field_only']['measurement']['sha256'])
    checkpoint=Path(evidence['field_only']['measurement']['path']).parent/field['checkpoint']['selection']['manifest']
    meta=selected(checkpoint,field['checkpoint']['selection']['manifest_sha256'])
    n1_bytes=meta['archive_bytes']+checkpoint.stat().st_size
    c1=log.decode((ROOT/'docs/evidence/NEURAL_MATERIAL_FIELD_VALIDATION_COMPLETION_V1.json').read_bytes())
    measured={r['arm']:r['archive_bytes']+r['manifest_bytes'] for r in c1['storage']}
    measured['N1']=max(measured['N1'],n1_bytes)
    # Four full state files per history: three scheduled plus one failure-only.
    # Round each measured archive/manifest pair up by its complete NPZ-header
    # and JSON catalog allowance (one extra measured pair, not seven scalars).
    provenance_allowance=len(log.canonical(original['sha256']))+32*(max(map(len,original['sha256']))+64+16)+4096
    state_caps={arm:2*size+provenance_allowance for arm,size in measured.items()}
    scheduled=9*3*sum(state_caps.values());failure_states=9*sum(state_caps.values())
    n1=original['initial_models']['M1/101/N1'];params=tensor_bytes(n1['parameters']);buffers=tensor_bytes(n1['buffers'])
    # Source audit: maximum external points336,512 diffuse/1024 specular;
    # diagnostic capture copies inputs, directions/PDF, return RGB/basis,
    # tapes, parameters and gradients. Three same-call model audits in initial
    # admission; at most one admission and27 endpoint diagnostics (overbound).
    per_point=3*(512+2*1024)+1024+(2*1024+1)+12*3*(512+1024)+32+9
    capture_one=4*336*per_point+2*params+buffers+512*(2*84+64)
    diagnostic_archive_cap=3*capture_one
    diagnostics=28*diagnostic_archive_cap
    journal_actual=full['reporting']['final_manifest']
    final=selected(journal_actual['path'],journal_actual['sha256'])
    journal_measured=final['journal_bytes']
    pretty_measured=log.decode((Path(full['measurement']['path']).parent/'reporting-result.json').read_bytes())
    report_path=Path(full['measurement']['path']).parent/'report.json'
    require(log.file_sha(report_path)==pretty_measured['report_file_sha256'],'D2 materialized report changed')
    # Four measured full-journal sizes permit float/string expansion and
    # admission variance. Add one maximum event per possible failure group.
    # This is an ENFORCED storage ceiling, not a claimed mathematical size
    # bound on arbitrary exception text. Runtime fails closed before its cap.
    journal_cap=4*journal_measured+28*log.MAX_RECORD_BYTES
    report_cap=4*report_path.stat().st_size
    controls=128*log.MAX_SUMMARY_BYTES  # IDs, anchors, authorization, process/manifest files.
    raw=scheduled+failure_states+diagnostics+journal_cap+2*report_cap+controls
    # One equal reserve for atomic temporaries, filesystem variability and
    # conservative projection error; no scientific timeout/memory relaxation.
    minimum=2*raw
    tensor_resident=sum(4*tensor_bytes(m['parameters'])+tensor_bytes(m['buffers']) for m in original['initial_models'].values())
    old=log.decode((ROOT/'docs/evidence/NEURAL_MATERIAL_FIELD_RECOVERY_V1.json').read_bytes())
    old_rss=old['process']['child_peak_RSS_bytes'];new_rss=full['peak_RSS_bytes'];field_rss=field['peak_RSS_bytes']
    memory_estimate=old_rss+new_rss+tensor_resident+2*state_caps['N1']+field_rss
    return dict(schema_version=1,provenance_manifest_allowance_bytes=provenance_allowance,additional_source_entries_cap=32,measured_checkpoint_pairs_bytes=measured,state_file_caps_bytes=state_caps,
        scheduled_states_bytes=scheduled,failure_only_states_bytes=failure_states,
        diagnostic_archive_cap_bytes=diagnostic_archive_cap,diagnostics_bytes=diagnostics,
        journal_measured_bytes=journal_measured,journal_cap_bytes=journal_cap,
        materialized_report_measured_bytes=report_path.stat().st_size,materialized_report_cap_bytes=report_cap,
        materialized_plus_atomic_temporary_bytes=2*report_cap,control_metadata_bytes=controls,
        subtotal_bytes=raw,safety_reserve_bytes=raw,minimum_free_bytes=minimum,
        capacity_scope='One run plus a separate same-filesystem offline report; enforced journal/diagnostic/report caps; safety reserve equal to subtotal',
        memory=dict(all27_parameter_gradient_two_moment_plus_buffers_bytes=tensor_resident,
            original_whole_execution_peak_RSS_bytes=old_rss,D2_full_reporting_peak_RSS_bytes=new_rss,
            N1_field_only_peak_RSS_bytes=field_rss,checkpoint_temporary_allowance_bytes=2*state_caps['N1'],
            conservative_accounted_RSS_bytes=memory_estimate,limit_bytes=4*1024**3,address_limit_bytes=8*1024**3,
            fits_accounting=memory_estimate<4*1024**3,
            guarantee=False,unmeasured='Simultaneous shader autograd/allocator/OS behavior. Sum double-counts measured base/runtime/models deliberately; original live RSS guard and8GiB AS backstop remain mandatory.'))


def capacity(parent,minimum):
    parent=Path(parent)
    require(parent.is_absolute() and parent.resolve()==parent and parent.is_dir() and parent.stat().st_uid==os.geteuid() and os.access(parent,os.W_OK|os.X_OK),'Owned unaliased approved parent required')
    require(not any((p/'.git').is_file() or (p/'.git/HEAD').is_file() for p in (parent,*parent.parents)),'Output parent inside checkout')
    for p in (ROOT,ROOT.parent/'data',ROOT.parent/'final_results'):
        require(parent!=p and p not in parent.parents and parent not in p.parents,'Output parent overlaps preserved inputs')
    available=shutil.disk_usage(parent).free;require(available>=minimum,'Insufficient conservative full-run capacity')
    return dict(parent=str(parent),available_bytes=available,minimum_free_bytes=minimum,allocated=False)
