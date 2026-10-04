"""Semantic challenge transforms and independently checkable annotations."""
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction as F
from .generator import (GroundedEvent, Observation,Unknown,Executed,Hypothetical,Denied,Forced,
                        generate,compile_history,_rng)
from .simulator import WorldSpec,Regime,Action,Variable as V,Step,simulate
from .inference import Predictive,Interventional,Counterfactual,ExactDistribution,evaluate
from .simulator import Clamp
from .language_catalog import render_catalog,TRAIN_NAMES,HOLDOUT_NAMES
from .journal import Message

class PairKind(StrEnum):
    PARAPHRASE='paraphrase'
    NEGATION='negation'
    MODE='mode'
    ORDER='order'
    TIME='time'
    BINDING='binding'
    KNOWNNESS='knownness'

class HoldoutStratum(StrEnum):
    TEMPLATES='new_templates'
    NAMES='new_names_compositions'

@dataclass(frozen=True)
class PairSide:
    events: tuple[GroundedEvent,...]
    messages: tuple[Message,...]

@dataclass(frozen=True)
class Invariant:
    checked_queries: int

@dataclass(frozen=True)
class StructuralOnly:
    checked_queries: int

@dataclass(frozen=True)
class DistinguishingWitness:
    query: Predictive | Interventional | Counterfactual
    tv: F
    checked_queries: int

@dataclass(frozen=True)
class ChallengePair:
    group_id: str
    seed: int
    kind: PairKind
    stratum: HoldoutStratum
    left: PairSide
    right: PairSide
    names: tuple[str,str]
    families: tuple[int,int]
    explicit_time: bool
    annotation: Invariant | StructuralOnly | DistinguishingWitness


def rebuild_events(spec,seed,episode,plan):
    """Resequence executed events, then regenerate all visible factual values."""
    time=0
    events=[]
    steps=[]
    for event in plan:
        if isinstance(event,Executed):
            time+=1
            steps.append(Step.command(event.object_id,event.action))
        elif isinstance(event,Forced):
            time+=1
            steps.append(Step((Action.HOLD,Action.HOLD),(Clamp(event.object_id,event.variable,event.value),)))
        events.append(replace(event,time=time))
    trace=simulate(spec,episode,seed,tuple(steps))
    return tuple(replace(e,value=trace.states[e.time][e.object_id].get(e.variable)) if type(e) is Observation else e for e in events)


def evaluation_queries(left,right):
    for obj in (0,1):
        yield Predictive(obj,(Step.hold(),),(V.M,V.Y))
    for c in (0,1):
        for obj in (0,1):
            yield Interventional(obj,(Clamp(obj,V.C,c),),(V.M,V.Y))
    # Common admissible physical references only; no question about a missing event.
    for obj in (0,1):
        for time,(a,b) in enumerate(zip(left.steps,right.steps),1):
            action=a.actions[obj]
            if action == b.actions[obj] and action in (Action.START,Action.STOP):
                yield Counterfactual(obj,time,Action.STOP if action == Action.START else Action.START,(V.M,V.Y))


def annotate(left_events,right_events,invariant=False):
    spec=WorldSpec.for_regime(Regime.DETERMINISTIC)
    left,right=compile_history(left_events),compile_history(right_events)
    checked=0
    for query in evaluation_queries(left,right):
        a,b=evaluate(spec,left,query),evaluate(spec,right,query)
        if not isinstance(a,ExactDistribution) or not isinstance(b,ExactDistribution):
            raise ValueError('pair contains impossible or undefined factual evidence')
        checked+=1
        tv=sum(abs(p-b.probability(x)) for x,p in a.probabilities)/2
        if invariant and tv:
            raise ValueError('claimed invariant changes causal answers')
        if not invariant and tv >= F(1,4):
            return DistinguishingWitness(query,tv,checked)
    return Invariant(checked) if invariant else StructuralOnly(checked)


def make_pair(seed,group_id,kind,stratum):
    if not isinstance(kind,PairKind) or not isinstance(stratum,HoldoutStratum):
        raise ValueError('typed kind and stratum required')
    spec=WorldSpec.for_regime(Regime.DETERMINISTIC)
    rng=_rng(seed,group_id,'challenge-plan')
    surface=_rng(seed,group_id,'challenge-surface')
    # Prefix diversity, not answer-dependent acceptance. Challenges use 8/12/16
    # messages; base train/validation/IID retain the registered 4/8/12/16 sampler.
    length=rng.choice((8,12,16))
    base=generate(spec,seed,group_id,length)
    obj=rng.randrange(2)
    explicit=kind not in (PairKind.ORDER,PairKind.NEGATION,PairKind.MODE)
    if kind == PairKind.PARAPHRASE:
        left=right=base.events
    elif kind == PairKind.TIME:
        plan=base.events[:-2]+(Executed(0,obj,Action.START),Executed(0,obj,Action.STOP))
        left=rebuild_events(spec,seed,group_id,plan)
        right=left[:-2]+(left[-1],left[-2])
    elif kind == PairKind.ORDER:
        prefix=base.events[:-2]
        left=rebuild_events(spec,seed,group_id,prefix+(Executed(0,obj,Action.START),Executed(0,obj,Action.STOP)))
        right=rebuild_events(spec,seed,group_id,prefix+(Executed(0,obj,Action.STOP),Executed(0,obj,Action.START)))
    elif kind in (PairKind.NEGATION,PairKind.MODE):
        prefix=base.events[:-2]+(Executed(0,obj,Action.START),)
        left=rebuild_events(spec,seed,group_id,prefix+(Executed(0,obj,Action.STOP),))
        variant=Denied if kind == PairKind.NEGATION else Hypothetical
        right=rebuild_events(spec,seed,group_id,prefix+(variant(0,obj,Action.STOP),))
    elif kind == PairKind.BINDING:
        # Two commands guarantee both sides have executed history even if the
        # base sampler's required command was in the removed suffix.
        prefix=base.events[:-2]+(Executed(0,obj,Action.STOP),)
        left=rebuild_events(spec,seed,group_id,prefix+(Executed(0,obj,Action.START),))
        right=rebuild_events(spec,seed,group_id,prefix+(Executed(0,1-obj,Action.START),))
    else:
        prefix=base.events[:-2]+(Executed(0,obj,Action.STOP),)
        left=rebuild_events(spec,seed,group_id,prefix+(Observation(0,obj,V.R,0),))
        right=rebuild_events(spec,seed,group_id,prefix+(Unknown(0,obj,V.R),))
    names=tuple(surface.sample(TRAIN_NAMES if stratum == HoldoutStratum.TEMPLATES else HOLDOUT_NAMES,2))
    family=surface.choice((2,3) if stratum == HoldoutStratum.TEMPLATES else (0,1))
    families=(family,(5-family if family >= 2 else 1-family) if kind == PairKind.PARAPHRASE else family)
    annotation=annotate(left,right,kind in (PairKind.PARAPHRASE,PairKind.TIME))
    return ChallengePair(group_id,seed,kind,stratum,PairSide(left,render_catalog(left,names,families[0],explicit)),
                         PairSide(right,render_catalog(right,names,families[1],explicit)),names,families,explicit,annotation)


def implicit_clock_valid(events):
    time=0
    for event in events:
        if isinstance(event,(Executed,Forced)):
            time+=1
        if event.time!=time:
            return False
    return True


def valid_transform(pair):
    left,right=pair.left.events,pair.right.events
    if len(left)!=len(right) or len(left) not in (8,12,16) or pair.left.messages==pair.right.messages:
        return False
    if not pair.explicit_time and (not implicit_clock_valid(left) or not implicit_clock_valid(right)):
        return False
    kind=pair.kind
    if kind==PairKind.PARAPHRASE:
        return left==right and pair.families[0]!=pair.families[1]
    if kind==PairKind.TIME:
        return pair.explicit_time and left[:-2]==right[:-2] and left[-2:]==tuple(reversed(right[-2:]))
    if kind==PairKind.ORDER:
        return (not pair.explicit_time and left[:-2]==right[:-2]
                and all(isinstance(e,Executed) for e in left[-2:]+right[-2:])
                and len({e.object_id for e in left[-2:]+right[-2:]})==1
                and tuple(e.action for e in left[-2:])==(Action.START,Action.STOP)
                and tuple(e.action for e in right[-2:])==(Action.STOP,Action.START))
    if left[:-1]!=right[:-1]:
        return False
    a,b=left[-1],right[-1]
    if kind in (PairKind.NEGATION,PairKind.MODE):
        return (isinstance(a,Executed) and isinstance(b,Denied if kind==PairKind.NEGATION else Hypothetical)
                and a.action==b.action==Action.STOP and a.object_id==b.object_id and a.time==b.time+1)
    if kind==PairKind.BINDING:
        return isinstance(a,Executed) and isinstance(b,Executed) and a.action==b.action and a.time==b.time and a.object_id!=b.object_id
    if kind==PairKind.KNOWNNESS:
        return type(a) is Observation and isinstance(b,Unknown) and a.value==0 and a.variable==b.variable==V.R and a.time==b.time and a.object_id==b.object_id
    return False


def verify_pair(pair):
    if not valid_transform(pair):
        return False
    spec=WorldSpec.for_regime(Regime.DETERMINISTIC)
    for index,side in enumerate((pair.left,pair.right)):
        history=compile_history(side.events)
        trace=simulate(spec,pair.group_id,pair.seed,history.steps)
        for event in side.events:
            if type(event) is Observation and event.value != trace.states[event.time][event.object_id].get(event.variable):
                return False
        if side.messages != render_catalog(side.events,pair.names,pair.families[index],pair.explicit_time):
            return False
    expected=annotate(pair.left.events,pair.right.events,pair.kind in (PairKind.PARAPHRASE,PairKind.TIME))
    return expected == pair.annotation
