"""Validated query-v1 wire subset for the two-device reference runtime."""
from .inference import Predictive, Interventional, Counterfactual
from .simulator import Action, Variable, Clamp, Step


def _obj(value):
    if value not in ('device-0','device-1'):
        raise ValueError('unknown object ID')
    return int(value[-1])


def decode_query_ast(q):
    common = {'schema_version','kind','object_id','outcome','horizon','evidence_ref'}
    extras = {'Predictive':{'future_actions'},'Interventional':{'interventions'},'Counterfactual':{'replaced_time','replacement','continuation_policy'}}
    if not isinstance(q,dict) or q.get('schema_version') != 'query-v1' or not isinstance(q.get('kind'),str) or q['kind'] not in extras or set(q) != common | extras[q['kind']]:
        raise ValueError('unexpected query fields or version')
    if type(q['horizon']) is not int or q['horizon'] < 0 or not isinstance(q['evidence_ref'],str) or not q['evidence_ref']:
        raise ValueError('invalid horizon or evidence reference')
    if not isinstance(q['outcome'],list):
        raise ValueError('outcome must be an array')
    outcome = tuple(Variable(v) for v in q['outcome'])
    obj = _obj(q['object_id'])
    if q['kind'] == 'Predictive':
        if not isinstance(q['future_actions'],list) or len(q['future_actions']) != q['horizon']:
            raise ValueError('future actions must match horizon')
        return Predictive(obj,tuple(Step.command(obj,Action(a)) for a in q['future_actions']),outcome)
    if q['kind'] == 'Interventional':
        if q['horizon'] != 1 or not isinstance(q['interventions'],list) or not q['interventions']:
            raise ValueError('nonempty hard interventions on the next slice required')
        clamps = []
        for c in q['interventions']:
            if not isinstance(c,dict) or set(c) != {'object_id','variable','value'}:
                raise ValueError('invalid hard intervention')
            clamps.append(Clamp(_obj(c['object_id']),Variable(c['variable']),c['value']))
        return Interventional(obj,tuple(clamps),outcome)
    if q['horizon'] != 0 or q['continuation_policy'] != 'preserve_factual_actions':
        raise ValueError('CF returns at factual horizon with preserved factual schedule')
    return Counterfactual(obj,q['replaced_time'],Action(q['replacement']),outcome)
