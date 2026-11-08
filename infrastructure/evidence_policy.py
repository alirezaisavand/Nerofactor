"""Versioned admission decisions; original observations are never rewritten."""
import json
from infrastructure.stage_sources import ROOT,PINS,digest
from infrastructure.paths import relative_path
from infrastructure.signatures import require

POLICY='dual-scientific-baseline-v1'
STATUSES={'accepted_for_selected_stage_baseline','valid_historical_or_diagnostic_only',
          'invalidated_for_selected_stage_baseline','awaiting_compatibility_review','unexecuted'}


def matrix():
    value=json.loads((ROOT/'docs/evidence/dual-baseline-supersession-v1.json').read_text())
    require(value['policy']==POLICY and value['selected_scientific_commits']==PINS,'Evidence policy/source conflict')
    require(all(row['status'] in STATUSES for row in value['entries']),'Unknown admission status')
    return value


def require_target_acceptance(evidence_id,stage,scope):
    rows=[row for row in matrix()['entries'] if row['id']==evidence_id]
    require(len(rows)==1,'Unknown/ambiguous evidence identity')
    row=rows[0]
    require(row['status']=='accepted_for_selected_stage_baseline' and row['stage']==stage and
            scope in row['accepted_scopes'],'Evidence is not admitted for this selected baseline/scope')
    for item in row['evidence']:
        require(item.get('root')=='repository','External native evidence requires separate reviewed admission')
        path=ROOT/relative_path(item['path'])
        require(path.resolve()==path and digest(path)==item['sha256'] and path.stat().st_size==item['bytes'],
                'Accepted evidence bytes changed')
    return row
