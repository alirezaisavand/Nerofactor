"""Data-only decoding of the reviewed Bell numeric NumPy pickle dialect.

No pickle.load/Unpickler, imports named by the payload, REDUCE calls or BUILD
methods are executed. Only two fixed-size floating matrices can be returned.
"""
import pickletools
import numpy as np
from infrastructure.signatures import require


class Symbol:
    def __init__(self,name):self.name=name


class Numeric:
    def __init__(self,kind,code=None):self.kind,self.code,self.value=kind,code,None


def decode_camera(data):
    require(isinstance(data,bytes) and 0<len(data)<=4096,'Bounded camera bytes required')
    stack=[];memo={};mark=object();count=0;stopped=False
    def tuple_mark():
        j=len(stack)-1
        while j>=0 and stack[j] is not mark:j-=1
        require(j>=0,'Missing numeric pickle marker');value=tuple(stack[j+1:]);del stack[j:];return value
    for op,arg,pos in pickletools.genops(data):
        count+=1;require(count<=256,'Camera opcode cap exceeded');name=op.name
        if name=='PROTO':require(arg==4,'Only reviewed protocol4 camera dialect admitted')
        elif name=='FRAME':require(arg<=4096,'Oversized pickle frame')
        elif name=='EMPTY_LIST':stack.append([])
        elif name=='MARK':stack.append(mark)
        elif name in ('SHORT_BINUNICODE','SHORT_BINBYTES','BININT1','BININT','BININT2'):stack.append(arg)
        elif name=='NONE':stack.append(None)
        elif name=='NEWFALSE':stack.append(False)
        elif name=='NEWTRUE':stack.append(True)
        elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
        elif name=='BINGET':require(arg in memo,'Invalid camera memo');stack.append(memo[arg])
        elif name=='STACK_GLOBAL':
            identifier=stack.pop();module=stack.pop();pair=(module,identifier)
            require(pair in (('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype')),'Forbidden camera global')
            stack.append(Symbol(pair))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            n=int(name[-1]);value=tuple(stack[-n:]);del stack[-n:];stack.append(value)
        elif name=='TUPLE':stack.append(tuple_mark())
        elif name=='REDUCE':
            args=stack.pop();symbol=stack.pop();require(type(symbol) is Symbol and type(args) is tuple,'No callable camera reducers')
            if symbol.name==('numpy','dtype'):
                require(args in (('f4',False,True),('f8',False,True)),'Only plain floating camera dtypes admitted')
                stack.append(Numeric('dtype',args[0]))
            else:
                require(symbol.name==('numpy.core.multiarray','_reconstruct') and len(args)==3 and type(args[0]) is Symbol and
                        args[0].name==('numpy','ndarray') and args[1:]==((0,),b'b'),'No arbitrary camera reconstruction')
                stack.append(Numeric('array'))
        elif name=='BUILD':
            state=stack.pop();target=stack[-1];require(type(target) is Numeric and type(state) is tuple,'No object BUILD methods')
            if target.kind=='dtype':
                require(state==(3,'<',None,None,None,-1,-1,0),'Only little-endian primitive camera dtype state admitted')
                target.value=np.dtype('<'+target.code)
            else:
                require(len(state)==5 and state[0]==1 and state[1] in ((3,4),(3,3)) and type(state[2]) is Numeric and
                        state[2].kind=='dtype' and state[2].value is not None and state[3] is False and type(state[4]) is bytes,
                        'Invalid camera ndarray data state')
                shape=state[1];dtype=state[2].value
                require(len(state[4])==int(np.prod(shape))*dtype.itemsize,'Camera byte/shape mismatch')
                target.value=np.frombuffer(state[4],dtype=dtype).copy().reshape(shape)
                require(np.isfinite(target.value).all(),'Nonfinite calibration')
        elif name=='APPENDS':
            items=tuple_mark();require(type(stack[-1]) is list and len(items)==2,'Expected exactly two calibration matrices');stack[-1].extend(items)
        elif name=='STOP':require(pos==len(data)-1,'Trailing camera bytes');stopped=True
        else:raise ValueError('Unsupported camera opcode: '+name)
    require(stopped and len(stack)==1 and type(stack[0]) is list and len(stack[0])==2,'Invalid camera container')
    require(all(type(v) is Numeric and v.kind=='array' and v.value is not None for v in stack[0]),'Camera must contain numeric matrices')
    pose,K=[v.value for v in stack[0]];require(pose.shape==(3,4) and K.shape==(3,3),'Camera matrix roles/shapes mismatch')
    return pose,K


def calibrations(paths,scale=128/800):
    pose=[];intrinsics=[]
    for p in paths:
        P,K=decode_camera(p.read_bytes());P=P.astype(np.float32);K=K.astype(np.float32)
        R=P[:,:3].astype(float)
        require(np.max(abs(R@R.T-np.eye(3)))<=2e-6 and abs(np.linalg.det(R)-1)<=2e-6,'Invalid camera rotation')
        require(np.max(abs(K[2]-[0,0,1]))<=1e-7 and K[0,0]>0 and K[1,1]>0 and abs(K[0,1])+abs(K[1,0])<1e-7,'Invalid camera intrinsics')
        pose.append(P);k=K.copy();k[:2]*=scale;intrinsics.append(k)
    return np.stack(pose),np.stack(intrinsics)


def rays(pose,K,height,width):
    y,x=np.mgrid[:height,:width];pixels=np.stack([x+.5,y+.5,np.ones_like(x)],-1).reshape(-1,3)
    directions=(pixels@np.linalg.inv(K.astype(float)).T)@pose[:,:3].astype(float)
    directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
    center=-pose[:,:3].astype(float).T@pose[:,3].astype(float)
    return np.broadcast_to(center,directions.shape).copy(),directions
