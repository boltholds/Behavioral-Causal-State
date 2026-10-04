"""Exact finite-state abduction/action/prediction using twin-state filtering.

Input is visible evidence and a fixed factual action schedule, never simulator U.
The known logging policy depends only on observed initial/current C, so its
likelihood cancels conditional on that C in SPEC-04. For other policies the
caller must supply a model with policy likelihoods; this evaluator does not
silently identify action conditioning with intervention.
"""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from .simulator import Action, Variable, Clamp, Step, DeviceState, bit, object_id, advance_device

@dataclass(frozen=True)
class Evidence:
    time: int
    object_id: int
    variable: Variable
    value: int
    def __post_init__(self):
        if type(self.time) is not int or self.time < 0:
            raise ValueError('nonnegative integer time required')
        object_id(self.object_id)
        bit(self.value)
        if not isinstance(self.variable,Variable):
            raise ValueError('Variable enum required')

@dataclass(frozen=True)
class History:
    steps: tuple[Step,...]
    evidence: tuple[Evidence,...]
    def __post_init__(self):
        if not isinstance(self.steps,tuple) or not all(isinstance(s,Step) for s in self.steps):
            raise ValueError('immutable steps required')
        if not isinstance(self.evidence,tuple) or not all(isinstance(e,Evidence) for e in self.evidence):
            raise ValueError('immutable evidence required')
        if any(e.time > len(self.steps) for e in self.evidence):
            raise ValueError('evidence outside factual horizon')


def _query(obj,outcome):
    object_id(obj)
    if not isinstance(outcome,tuple) or not outcome or len(set(outcome)) != len(outcome) or not all(isinstance(v,Variable) for v in outcome):
        raise ValueError('nonempty distinct typed outcome variables required')

@dataclass(frozen=True)
class Predictive:
    object_id: int
    future: tuple[Step,...]
    outcome: tuple[Variable,...]
    def __post_init__(self):
        _query(self.object_id,self.outcome)
        if not isinstance(self.future,tuple) or not all(isinstance(s,Step) for s in self.future):
            raise ValueError('immutable future steps required')

@dataclass(frozen=True)
class Interventional:
    object_id: int
    clamps: tuple[Clamp,...]
    outcome: tuple[Variable,...]
    def __post_init__(self):
        _query(self.object_id,self.outcome)
        Step((Action.HOLD,Action.HOLD),self.clamps)

@dataclass(frozen=True)
class Counterfactual:
    object_id: int
    replaced_time: int
    replacement: Action
    outcome: tuple[Variable,...]
    def __post_init__(self):
        _query(self.object_id,self.outcome)
        if type(self.replaced_time) is not int or self.replaced_time < 1 or self.replacement not in (Action.START,Action.STOP) or not isinstance(self.replacement,Action):
            raise ValueError('replace a past start/stop by a typed command')

@dataclass(frozen=True)
class ExactDistribution:
    probabilities: tuple[tuple[tuple[int,...],F], ...]
    evidence_probability: F
    transitions: int
    def probability(self,outcome):
        return dict(self.probabilities).get(outcome,F(0))

@dataclass(frozen=True)
class InferenceUndefined:
    reason: str

@dataclass(frozen=True)
class InferenceIncomplete:
    reason: str
    transitions: int

class _BudgetExhausted(Exception):
    pass


def evaluate(spec,history,query,budget=1_000_000):
    """Distribution conditional on the declared SCM, not identification over SCMs.

    Normalizing after each slice retains the likelihood separately. Both devices
    are checked, including evidence on an object other than the query target.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError('nonnegative transition budget required')
    if not isinstance(query,(Predictive,Interventional,Counterfactual)):
        raise ValueError('unsupported query')
    if isinstance(query,Counterfactual):
        t = query.replaced_time
        if t > len(history.steps) or history.steps[t-1].actions[query.object_id] not in (Action.START,Action.STOP):
            raise ValueError('CF reference must point to an executed start/stop')
        if history.steps[t-1].actions[query.object_id] == query.replacement:
            raise ValueError('CF must replace start by stop or stop by start')
    future = query.future if isinstance(query,Predictive) else (Step((Action.HOLD,Action.HOLD),query.clamps),) if isinstance(query,Interventional) else ()
    count = 0
    evidence_probability = F(1)
    target = {}

    def spend():
        nonlocal count
        if count >= budget:
            raise _BudgetExhausted
        count += 1

    try:
        for obj in range(2):
            evidence = defaultdict(list)
            for e in history.evidence:
                if e.object_id == obj:
                    evidence[e.time].append(e)
            def consistent(state,time):
                return all(state.get(e.variable) == e.value for e in evidence[time])
            pairs = defaultdict(F)
            for c in (0,1):
                for nm,ny,p in spec.noise.support():
                    spend()
                    state = advance_device(DeviceState(0,0,c,0),Action.HOLD,(),nm,ny,spec.wrong_gate)
                    if consistent(state,0):
                        pairs[(state,state)] += p/2
            mass = sum(pairs.values(),F(0))
            if not mass:
                return InferenceUndefined('zero-support factual evidence')
            evidence_probability *= mass
            pairs = {pair:p/mass for pair,p in pairs.items()}
            for time,step in enumerate(history.steps,1):
                next_pairs = defaultdict(F)
                alt = query.replacement if isinstance(query,Counterfactual) and obj == query.object_id and time == query.replaced_time else step.actions[obj]
                clamps = step.local_clamps(obj)
                for (factual,alternative),weight in pairs.items():
                    for nm,ny,p in spec.noise.support():
                        spend()
                        f = advance_device(factual,step.actions[obj],clamps,nm,ny,spec.wrong_gate)
                        a = advance_device(alternative,alt,clamps,nm,ny,spec.wrong_gate)
                        if consistent(f,time):
                            next_pairs[(f,a)] += weight*p
                mass = sum(next_pairs.values(),F(0))
                if not mass:
                    return InferenceUndefined('zero-support factual evidence')
                evidence_probability *= mass
                pairs = {pair:p/mass for pair,p in next_pairs.items()}
            if obj == query.object_id:
                for (_,alternative),p in pairs.items():
                    target[alternative] = target.get(alternative,F(0)) + p
        for step in future:
            nxt = defaultdict(F)
            obj = query.object_id
            for state,weight in target.items():
                for nm,ny,p in spec.noise.support():
                    spend()
                    nxt[advance_device(state,step.actions[obj],step.local_clamps(obj),nm,ny,spec.wrong_gate)] += weight*p
            target = nxt
    except _BudgetExhausted:
        return InferenceIncomplete('transition budget exhausted; no partial distribution asserted',count)
    probabilities = {outcome:F(0) for outcome in product((0,1),repeat=len(query.outcome))}
    for state,p in target.items():
        probabilities[tuple(state.get(v) for v in query.outcome)] += p
    return ExactDistribution(tuple(probabilities.items()),evidence_probability,count)
