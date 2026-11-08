"""Read-only Python-call observation around original G initialization/validation.

No renderer/dataset/metric method is replaced and no RNG is drawn. Returned
validation tensors are retained, not altered, for lossless export after evaluation.
"""
from pathlib import Path
import inspect
import sys
from infrastructure.signatures import require


def description(value):
    if type(value) in (tuple,list): return [description(x) for x in value]
    if type(value) is dict: return {str(k):description(v) for k,v in value.items()}
    shape=getattr(value,'shape',None)
    return {'shape':list(shape),'dtype':str(getattr(value,'dtype',None))} if shape is not None else {'type':type(value).__name__}


class ReadAudit:
    def __init__(self,database_class,renderer_class,build_info,split,read_pickle,forbidden=(),loader=None,progress_class=None,allow_forward=True):
        self.getters={getattr(database_class,n).__code__:n for n in
                      ['get_image','get_pose','get_K','get_mask','get_depth','get_img_ids']}
        self.forward=renderer_class.forward.__code__;self.build=build_info.__code__;self.split=split.__code__
        self.pickle=read_pickle.__code__
        self.forbidden={inspect.unwrap(f).__code__ for f in forbidden if hasattr(inspect.unwrap(f),'__code__')}
        self.phase='initialization';self.allow_forward=allow_forward
        self.loader=loader;self.progress=progress_class
        self.loader_code=type(loader).__iter__.__code__ if loader is not None else None
        self.progress_code=getattr(getattr(progress_class,'__init__',None),'__code__',None)
        self.reads=[];self.builds=[];self.splits=[];self.pickles=[];self.forwards=[];self.outputs=[]
        self.databases=[];self.iterators=[];self.bars=[];self.cleanup_errors=[];self.started=False

    def database_id(self,value):
        for i,obj in enumerate(self.databases):
            if obj is value:return 'database-'+str(i+1)
        self.databases.append(value);return 'database-'+str(len(self.databases))

    def watch_loader(self,loader,progress_class):
        self.loader=loader;self.loader_code=type(loader).__iter__.__code__
        self.progress=progress_class;self.progress_code=progress_class.__init__.__code__

    def observe(self,frame,event,value):
        code=frame.f_code;v=frame.f_locals
        if event=='return' and value is None:return  # Preserve original exception tracebacks.
        if event=='call' and code in self.forbidden:
            require(False,'Forbidden update/training/checkpoint writer reached during read-only diagnostic')
        if code in self.getters and event=='return':
            method=self.getters[code]
            row={'phase':self.phase,'database':self.database_id(v['self']),'method':method,
                 'source_id':str(v['img_id']) if 'img_id' in v else None,'returned':description(value)}
            if method=='get_img_ids':row['ordered_ids']=[str(i) for i in value]
            self.reads.append(row)
        elif code is self.pickle and event=='call':
            name=code.co_varnames[0];path=v.get(name)
            require(isinstance(path,(str,Path)),'Unexpected original camera/cache read argument')
            self.pickles.append({'phase':self.phase,'path':str(path),'resolved':str(Path(path).resolve())})
        elif code is self.build and event=='return':
            self.builds.append({'database':self.database_id(v['database']),'source_ids':[str(i) for i in v['img_ids']],
                                'returned':description(value)})
        elif code is self.split and event=='return':
            require(type(value) in (list,tuple) and len(value)==2,'G split return contract changed')
            self.splits.append({'database':self.database_id(v['database']),'kind':v.get('split_type','validation'),
                                'train_ids':[str(i) for i in value[0]],'test_ids':[str(i) for i in value[1]]})
        elif code is self.forward:
            if event=='call':
                require(self.allow_forward and self.phase=='evaluation' and not self.forwards,'Exactly one explicit G validation forward allowed')
                data=v['data']
                require(set(data)=={'index','eval','step'} and type(data['index']) is int and data['index']==0 and
                        data['eval'] is True and type(data['step']) is int and data['step']==149999,'Unexpected G validation dispatch')
                self.forwards.append(dict(data))
            elif event=='return':
                require(type(value) is dict,'Original renderer failed to return validation outputs')
                self.outputs.append(value)
        elif code is self.loader_code and event=='return' and v.get('self') is self.loader:
            self.iterators.append(value)
        elif code is self.progress_code and event=='call' and self.phase=='evaluation':
            self.bars.append(v['self'])

    def __enter__(self):
        require(not self.started and sys.implementation.name=='cpython' and sys.getprofile() is None and sys.gettrace() is None,
                'Unprofiled/untraced CPython required for read observation')
        self.started=True;sys.setprofile(self.observe)
        return self

    def __exit__(self,*args):
        sys.setprofile(None)
        for kind,items,method in [('loader',self.iterators,'_shutdown_workers'),('progress',self.bars,'close')]:
            for item in items:
                if callable(getattr(item,method,None)):
                    try:getattr(item,method)()
                    except Exception as exc:self.cleanup_errors.append({'resource':kind,'error':str(exc)})
        if args[0] is None:require(not self.cleanup_errors,'Validation resource cleanup failed')

    def record(self,model):
        key=self.database_id(model.database)
        require([str(i) for i in model.train_ids]==[str(i) for i in range(1,128)] and list(model.test_ids)==['0'],
                'Original G renderer selected different ordered IDs')
        selected=[b for b in self.builds if b['database']==key and b['source_ids']==['0']]
        require(len(selected)==1,'Original source-0 validation image-info build not observed')
        require(len(self.forwards)==len(self.outputs)==1,'Missing/surplus original validation forward')
        depth=[r for r in self.reads if r['phase']=='evaluation' and r['method']=='get_depth']
        require(len(depth)==1 and depth[0]['database']==key and depth[0]['source_id']=='0','Validation index 0 was not source camera 0')
        for method in ['get_image','get_pose','get_K','get_mask']:
            require(any(r['database']==key and r['method']==method and r['source_id']=='0' for r in self.reads),
                    'Source-0 original read missing: '+method)
        expected={'train_ids':[str(i) for i in range(1,128)],'test_ids':['0']}
        require(len(self.splits)==2 and all(s['kind']=='validation' and all(s[k]==v for k,v in expected.items()) for s in self.splits),
                'Original validation-loader/renderer G split call sequence changed')
        return {'model_database':key,'database_source_id':'0','validation_iterator_index':0,
                'original_G_splits':expected,'split_calls':self.splits,'getter_reads':self.reads,
                'image_info_builds':self.builds,'camera_pickle_reads':self.pickles,'forward_calls':self.forwards,
                'cleanup_errors':self.cleanup_errors,'observation':'Original method calls observed, not replaced; no raw tensor values in read summaries.'}
