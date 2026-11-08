"""Frozen G-only JSON/tensor-hash gates. No A reference or numerical tolerance reuse."""
from pathlib import Path
from infrastructure.g_training import PROTOCOL
from infrastructure.stage_sources import PINS,baseline,digest,ROOT,infrastructure_identity
from infrastructure.signatures import require,read_report
from infrastructure.bell_comparison import artifact


def validate(report,arm):
    _,source=baseline('geometry',PINS['geometry'])
    require(report.get('schema_version')==1 and report.get('protocol')==PROTOCOL and report.get('arm')==arm and
            report.get('stage')=='geometry' and report.get('scientific_commit')==PINS['geometry'] and
            report.get('status')=='complete' and report.get('phase')=='bounded_stop_after_second_optimizer_return',
            'Not a completed G geometry training arm; A evidence cannot be substituted')
    context=report.get('source',{})
    require(context.get('stage')=='geometry' and context.get('scientific_commit')==PINS['geometry'] and
            context.get('algorithm')=='geometry-G' and context.get('scientific_files')==source['files'] and
            context.get('source_manifest_sha256')==digest(ROOT/'scientific_sources/manifest-v1.json'),
            'Unverified G scientific source provenance')
    require(context.get('infrastructure_files')==infrastructure_identity(),
            'Infrastructure/protocol hashes changed since G execution')
    require(report.get('optimizer_calls_completed')==2 and report.get('checkpoint_input') is None and
            report.get('cleanup_errors')==[],'G requires fresh two-update success with cleanup')
    require(report.get('splits')=={'train_ids':[str(i) for i in range(1,128)],'test_ids':['0'],'nvs_ids':None},'Wrong G split')
    require(report.get('clipping',{}).get('enabled') is False,'Clipped geometry evidence is not G')
    require(report.get('route')==('schema-1-pinned-stage' if arm=='C' else 'independent-pinned-legacy-FullLoader'),'Wrong configuration route')
    require(report['initial_optimizer']['state']=={} and report['initial_optimizer']['groups'][0]['eps']==1e-8 and
            report['initial_optimizer']['groups'][0]['amsgrad'] is False,'Not G Adam')
    require(report['scheduler']['attributes']['learning_rate_alpha']==0.05 and
            report['scheduler']['attributes']['end_iter']==300000,'Not G scheduler')
    require(len(report['updates'])==2,'Incomplete G update sequence')
    previous=report['initial_parameters']
    for i,u in enumerate(report['updates']):
        require(type(u['step']) is int and u['step']==i and u['update_index']==i+1 and
                u['parameters_before']==previous and u['clipping_applied'] is False,'Wrong update chain/clipping')
        require(u['lr']==(0.0,5e-4*(1/5000))[i] and u['cursor_after']==(i+1)*512,'Wrong verified G warm-up/ray cursor')
        require(u['gradient_l2_before_optimizer_cpu64']>0,'Missing G gradients')
        require((i==0 and u['changed_elements']==0) or (i==1 and u['changed_elements']>0),'G parameter updates failed')
        require(all(set(v)=={'step','exp_avg','exp_avg_sq'} for v in u['optimizer']['state'].values()),'G Adam state has AMSGrad/unknown fields')
        previous=u['parameters_after']
    return report


def load(path,digest,arm):
    report,identity=read_report(path,digest);validate(report,arm)
    require(report.get('output_hashes'),'Missing G output manifest')
    for name in report['output_hashes']:artifact(Path(path).resolve().parent,name,report['output_hashes'])
    return report,identity


def pair(a,b):
    fields=['stage','scientific_commit','source','declared_image_id','legacy_sha256','original_configuration',
            'runtime_configuration','input_files','initial_parameters','initial_buffers','initial_optimizer',
            'effective_trainer','effective_renderer','effective_shader','effective_losses','splits','clipping','scheduler','loader','runtime',
            'imported_scientific_modules']
    mismatch=[k for k in fields if a[k]!=b[k]]
    require(a['run_id']!=b['run_id'],'Independent fresh G runs required')
    # Timing/memory instrumentation are not scientific repeatability criteria.
    evidence=['step','update_index','lr','losses','aggregate_loss','gradients_before_optimizer',
              'gradient_l2_before_optimizer_cpu64','clipping_applied','parameters_before','parameters_after',
              'parameter_changes','changed_elements','optimizer','sampling','cursor_after',
              'rng_before_batch','rng_before_forward','rng_after']
    differences={str(i):[k for k in evidence if x[k]!=y[k]] for i,(x,y) in enumerate(zip(a['updates'],b['updates']))}
    return {'passed':not mismatch and not any(differences.values()),'identity_mismatches':mismatch,'update_mismatches':differences,
            'policy':'exact metadata/scalar equality and exact tensor/state hashes; no borrowed A or RGB tolerance'}


def sequence(r,repeat,c=None):
    validate(r,'R');validate(repeat,'R-repeat')
    gate=pair(r,repeat)
    result={'schema_version':1,'protocol':PROTOCOL,'repeatability':gate,'candidate':None,'result':'G_repeatability_failed'}
    if not gate['passed']:return result
    result['result']='G_repeatability_passed'
    if c is not None:
        validate(c,'C');result['candidate']=pair(r,c)
        result['result']='G_regression_passed' if result['candidate']['passed'] else 'G_candidate_mismatch'
    return result
