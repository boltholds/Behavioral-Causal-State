"""Strict evaluator sidecar codecs; never passed to a text learner."""
from dataclasses import asdict
from fractions import Fraction as F
from .generator import Observation,Unknown,Forced,Executed,Hypothetical,Denied,event_wire
from .simulator import Variable as V,Action,Clamp,Step
from .inference import History,Evidence,Predictive,Interventional,Counterfactual
from .calibration import CalibrationProbe,CalibrationPanel
from .pairs import Invariant,StructuralOnly,DistinguishingWitness
from .queries import decode_query_ast


def event_from_wire(value):
    if not isinstance(value,dict) or set(value) != {'time','entity','mode','predicate','negation','value'}:
        raise ValueError('invalid event fields')
    if value['entity'] not in ('device-0','device-1'):
        raise ValueError('invalid entity')
    obj=int(value['entity'][-1]); time=value['time']; mode=value['mode']
    command={'Executed':Executed,'Hypothetical':Hypothetical,'Denied':Denied}
    if mode in command:
        if value['predicate'] != 'command' or value['negation'] != ('execution' if mode=='Denied' else 'none'):
            raise ValueError('invalid command semantics')
        event=command[mode](time,obj,Action(value['value']))
    else:
        if value['negation'] != 'none':
            raise ValueError('invalid value negation')
        var=V(value['predicate'])
        if mode=='Unknown' and value['value']=={'kind':'Unknown'}:
            event=Unknown(time,obj,var)
        elif mode in ('Observation','Forced') and isinstance(value['value'],dict) and set(value['value'])=={'kind','value'} and value['value']['kind']=='Binary':
            event=(Observation if mode=='Observation' else Forced)(time,obj,var,value['value']['value'])
        else:
            raise ValueError('invalid event variant')
    return event


def query_wire(query):
    q={'schema_version':'query-v1','kind':type(query).__name__,'object_id':f'device-{query.object_id}',
       'outcome':[v.value for v in query.outcome],'evidence_ref':'current_history'}
    if isinstance(query,Predictive):
        return q|{'horizon':len(query.future),'future_actions':[s.actions[query.object_id].value for s in query.future]}
    if isinstance(query,Interventional):
        return q|{'horizon':1,'interventions':[{'object_id':f'device-{c.object_id}','variable':c.variable.value,'value':c.value} for c in query.clamps]}
    return q|{'horizon':0,'replaced_time':query.replaced_time,'replacement':query.replacement.value,'continuation_policy':'preserve_factual_actions'}


def annotation_wire(annotation):
    result={'kind':type(annotation).__name__,'checked_queries':annotation.checked_queries}
    if isinstance(annotation,DistinguishingWitness):
        result|={'query':query_wire(annotation.query),'tv':str(annotation.tv)}
    return result


def annotation_from_wire(value):
    n=value['checked_queries']
    if type(n) is not int or n < 1:
        raise ValueError('invalid query count')
    if value['kind']=='DistinguishingWitness' and set(value)=={'kind','checked_queries','query','tv'}:
        tv=F(value['tv'])
        if not F(1,4) <= tv <= 1:
            raise ValueError('invalid witness TV')
        return DistinguishingWitness(decode_query_ast(value['query']),tv,n)
    if set(value)=={'kind','checked_queries'} and value['kind'] in ('Invariant','StructuralOnly'):
        return (Invariant if value['kind']=='Invariant' else StructuralOnly)(n)
    raise ValueError('invalid annotation')


def panel_wire(panel):
    return {'schema':'calibration-panel-v1','action_count':panel.action_count,'probes':[asdict(p) for p in panel.probes]}


def panel_from_wire(value):
    if set(value) != {'schema','action_count','probes'} or value['schema'] != 'calibration-panel-v1':
        raise ValueError('invalid panel schema')
    probes=[]
    for p in value['probes']:
        steps=tuple(Step(tuple(Action(a) for a in s['actions']),tuple(Clamp(c['object_id'],V(c['variable']),c['value']) for c in s['clamps'])) for s in p['history']['steps'])
        evidence=tuple(Evidence(e['time'],e['object_id'],V(e['variable']),e['value']) for e in p['history']['evidence'])
        probes.append(CalibrationProbe(V(p['target']),History(steps,evidence),p['output_time'],p['object_id']))
    panel=CalibrationPanel(tuple(probes))
    if panel.action_count != value['action_count']:
        raise ValueError('incorrect action accounting')
    return panel
