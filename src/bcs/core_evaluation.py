"""Frozen core evaluation; evaluator data never changes learned tables."""
from itertools import product,combinations
from pathlib import Path
import json
from statistics import mean
from .calibration import BooleanCore
from .dynamics import Dynamics
from .dataset_codec import event_from_wire
from .generator import compile_history,Executed
from .simulator import WorldSpec,Regime,DeviceState,Action,Variable as V,Step,Clamp
from .inference import Predictive,Interventional,Counterfactual,ExactDistribution,evaluate
from .metrics import empirical_bernstein_upper
from .dataset import audit_dataset


def compare_kernels(core: Dynamics,reference: Dynamics):
    checked=mismatches=0
    for c in (0,1):
        mismatches+=core.reset(c,0,0)!=reference.reset(c,0,0)
    for values in product((0,1),repeat=4):
        previous=DeviceState(*values)
        for action in Action:
            for assignments in product((-1,0,1),repeat=4):
                clamps=tuple((var,value) for var,value in zip(V,assignments) if value!=-1)
                checked+=1
                mismatches+=core.advance(previous,action,clamps,0,0)!=reference.advance(previous,action,clamps,0,0)
    return {'checked_transitions':checked,'checked_resets':2,'mismatches':mismatches,'scope':'exhaustive local deterministic kernel, known graph and declared reset'}


def _tv(core,spec,history,query):
    a,b=evaluate(core,history,query),evaluate(spec,history,query)
    if not isinstance(a,ExactDistribution) or not isinstance(b,ExactDistribution):
        return 1.0
    return float(sum(abs(p-b.probability(x)) for x,p in a.probabilities)/2)


def evaluate_core(dataset,progress=lambda value:None):
    """Run artifact audit before any validation claim, including direct API calls."""
    dataset=Path(dataset)
    audit=audit_dataset(dataset,progress)
    if audit['integrity_status']!='passed':
        return {'oracle_validation_gate':'invalid_dataset','audit_report':audit,
                'L1_status':'not_run_neural_branches','test_feedback_to_training':False}
    core=BooleanCore.from_wire(json.loads((dataset/'core.json').read_text()))
    manifest=json.loads((dataset/'manifest.json').read_text())
    spec=WorldSpec.for_regime(Regime.DETERMINISTIC)
    categories={k:[] for k in ('predict_hold','do_c_zero_hold','do_c_one_hold','replace_past_command')}
    joint=[]
    for index,line in enumerate((dataset/'evaluator/validation.jsonl').open()):
        row=json.loads(line)
        events=tuple(event_from_wire(e) for e in row['sides'][0]['events'])
        history=compile_history(events)
        categories['predict_hold'].append(mean(_tv(core,spec,history,Predictive(obj,(Step.hold(),),(V.M,V.Y))) for obj in (0,1)))
        for c,key in ((0,'do_c_zero_hold'),(1,'do_c_one_hold')):
            categories[key].append(mean(_tv(core,spec,history,Interventional(obj,(Clamp(obj,V.C,c),),(V.M,V.Y))) for obj in (0,1)))
        cf=[Counterfactual(e.object_id,e.time,Action.STOP if e.action==Action.START else Action.START,(V.M,V.Y)) for e in events if isinstance(e,Executed)]
        if not cf:
            raise ValueError('validation history lacks eligible CF reference')
        categories['replace_past_command'].append(mean(_tv(core,spec,history,q) for q in cf))
        joint.append(mean(_tv(core,spec,history,Interventional(obj,tuple(Clamp(obj,v,x) for v,x in zip(targets,assignment)),(V.M,V.Y))) for obj in (0,1) for targets in combinations((V.R,V.M,V.C),2) for assignment in product((0,1),repeat=2)))
        if (index+1)%500==0:
            progress({'core_validation_histories':index+1})
    if len(joint)<2:
        raise ValueError('at least two independent histories required for mean bound')
    language=[mean(values) for values in zip(*categories.values())]
    lang_upper=empirical_bernstein_upper(language,0,1)
    joint_upper=empirical_bernstein_upper(joint,0,1)
    kernel=compare_kernels(core,spec)
    full=manifest['mode']=='registered_size' and len(joint)==4000
    return {'core_artifact_id':core.artifact_id,'data_split':'validation','histories':len(joint),
            'category_means':{k:mean(v) for k,v in categories.items()},'language_mean_tv':mean(language),'joint_mean_tv':mean(joint),
            'language_tv_upper':lang_upper,'joint_tv_upper':joint_upper,'kernel':kernel,
            'oracle_validation_gate':('passed' if max(lang_upper,joint_upper)<=.02 and kernel['mismatches']==0 else 'failed') if full else 'not_run_registered_dataset',
            'audit_report':audit,
            'L1_status':'not_run_neural_branches','test_feedback_to_training':False}
