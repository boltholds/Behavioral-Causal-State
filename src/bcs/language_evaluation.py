"""Free-running structural and causal diagnostics; never supplies model inputs."""
from collections import Counter
from itertools import combinations, product
from statistics import mean
from .language_tokens import FIELDS, EOS, decode_events, InvalidDecoding
from .generator import compile_history, Executed
from .inference import evaluate, ExactDistribution, Predictive, Interventional, Counterfactual
from .simulator import Action, Step, Clamp, Variable as V
from .metrics import empirical_bernstein_upper


def structural_report(predictions, targets):
    if len(predictions) != len(targets) or not targets:
        raise ValueError('nonempty aligned predictions required')
    confusion = {f: Counter() for f in FIELDS}
    wrong_events = total_events = wrong_histories = 0
    for predicted, target in zip(predictions, targets):
        predicted, target = tuple(predicted), tuple(target)
        wrong_histories += predicted != target
        p = predicted[:-1] if predicted and predicted[-1] == EOS else predicted
        t = target[:-1]
        count = max((len(p)+5)//6, (len(t)+5)//6)
        for i in range(count):
            total_events += 1
            wrong = False
            for j, field in enumerate(FIELDS):
                index = 6*i+j
                actual = p[index] if index < len(p) else -1
                expected = t[index] if index < len(t) else -1
                confusion[field][f'{expected}->{actual}'] += 1
                wrong |= expected != actual
            wrong_events += wrong
    fields = {}
    for field, counts in confusion.items():
        errors = sum(n for pair,n in counts.items() if len(set(pair.split('->'))) != 1)
        fields[field] = {'errors': errors, 'total': total_events, 'error_rate': errors/total_events, 'confusion': dict(counts)}
    return {'histories': len(targets), 'full_history_error': wrong_histories/len(targets),
            'full_event_error': wrong_events/total_events, 'fields': fields,
            'invalid_decodings': sum(isinstance(decode_events(p),InvalidDecoding) for p in predictions)}


def causal_report(predictions, true_events, core):
    if len(predictions) != len(true_events) or not true_events:
        raise ValueError('nonempty aligned histories required')
    keys = ('predict_hold','do_c_zero_hold','do_c_one_hold','replace_past_command')
    categories = {k: [] for k in keys}; joint=[]; invalid=0; undefined=0; query_count=0
    for tokens, events in zip(predictions,true_events):
        decoded=decode_events(tokens); truth=compile_history(events)
        if isinstance(decoded,InvalidDecoding):
            invalid+=1
            for values in categories.values():values.append(1.0)
            joint.append(1.0)
            continue
        predicted=compile_history(decoded.events)
        def distance(query):
            nonlocal undefined,query_count
            query_count+=1
            expected=evaluate(core,truth,query)
            if not isinstance(expected,ExactDistribution):
                raise ValueError('gold query is not defined under frozen core')
            try: answer=evaluate(core,predicted,query)
            except ValueError: answer=False
            if not isinstance(answer,ExactDistribution):
                undefined+=1;return 1.0
            p=dict(answer.probabilities);q=dict(expected.probabilities)
            return float(sum(abs(p.get(x,0)-q.get(x,0)) for x in p.keys()|q.keys())/2)
        categories[keys[0]].append(mean(distance(Predictive(obj,(Step.hold(),),(V.M,V.Y))) for obj in (0,1)))
        for c in (0,1):
            categories[keys[c+1]].append(mean(distance(Interventional(obj,(Clamp(obj,V.C,c),),(V.M,V.Y))) for obj in (0,1)))
        queries=[Counterfactual(e.object_id,e.time,Action.STOP if e.action==Action.START else Action.START,(V.M,V.Y)) for e in events if isinstance(e,Executed)]
        if not queries:raise ValueError('gold history lacks eligible CF reference')
        categories[keys[3]].append(mean(distance(q) for q in queries))
        joint.append(mean(distance(Interventional(obj,tuple(Clamp(obj,v,x) for v,x in zip(vs,xs)),(V.M,V.Y))) for obj in (0,1) for vs in combinations((V.R,V.M,V.C),2) for xs in product((0,1),repeat=2)))
    language=[mean(row) for row in zip(*categories.values())]
    means={k:mean(v) for k,v in categories.items()}
    result={'category_means':means,'language_mean_tv':mean(language),'joint_mean_tv':mean(joint),
            'invalid_histories':invalid,'undefined_queries_on_decodable_histories':undefined,'queries_on_decodable_histories':query_count,
            'core_artifact_id':core.artifact_id,'scope':'diagnostic oracle comparison; no registered L1 acceptance',
            'sensitivity':{name:sum(means[k]*w for k,w in zip(keys,weights)) for name,weights in (('intervention_priority',(.1,.35,.35,.2)),('counterfactual_priority',(.1,.2,.2,.5)))}}
    if len(language)>=2:
        result.update(language_tv_upper=empirical_bernstein_upper(language,0,1),joint_tv_upper=empirical_bernstein_upper(joint,0,1))
    return result
