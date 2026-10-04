"""P1: verified response-type witnesses, explicit inner ranges, analytic bounds."""
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction as F
from time import perf_counter
import math
import numpy as np
from .contracts import CertifiedBounds, Undefined, Scope, WitnessRange, Identification
from .misspecification import Interval

@dataclass(frozen=True)
class ResponseModel:
    t: F
    def __post_init__(self):
        if not isinstance(self.t,F) or not 0 <= self.t <= F(1,4):
            raise ValueError('exact feasible t in [0,1/4] required')
    @property
    def weights(self):
        return self.t,F(3,4)-self.t,F(1,4)-self.t,self.t
    def marginals(self):
        w = self.weights
        return w[2]+w[3],w[1]+w[3]
    def query(self):
        return 4*self.t


def response_bounds(p0: Interval,p1: Interval,monotone=False):
    if p0.lower == 0:
        return Undefined('conditioning mass may be zero; domain policy required')
    if monotone:
        if p0.lower > p1.upper:
            return Undefined('empty class under monotonicity')
        return CertifiedBounds(1.0,1.0)
    lower = max(F(0),1-(1-p1.lower)/p0.lower)
    upper = min(F(1),p1.upper/p0.lower)
    lo,hi = float(lower),float(upper)
    if F(lo) > lower:
        lo = math.nextafter(lo,-math.inf)
    if F(hi) < upper:
        hi = math.nextafter(hi,math.inf)
    return CertifiedBounds(lo,hi)

class SearchArm(StrEnum):
    FIT = 'fit_only'
    NOVELTY = 'novelty'
    DIRECTED = 'directed'
    UNIFORM = 'uniform'

@dataclass(frozen=True)
class SearchResult:
    arm: SearchArm
    seed: int
    evaluations: int
    feasibility_checks: int
    result: WitnessRange
    identification: Identification
    scope: Scope
    trajectory: tuple[float,...]
    first_both_endpoints: int | str
    endpoint_t: tuple[float,float]
    wall_seconds: float


def _farthest(values,ids,n):
    # IDs globally increasing; stable smallest-ID ties including duplicates.
    selected = [int(np.argmin(ids))]
    distances = np.abs(values-values[selected[0]])
    distances[selected[0]] = -1
    for _ in range(n-1):
        candidates = np.flatnonzero(distances == distances.max())
        index = int(candidates[np.argmin(ids[candidates])])
        selected.append(index)
        distances = np.minimum(distances,np.abs(values-values[index]))
        distances[selected] = -1
    return np.array(selected)


def search(arm,seed,generations=100,population_size=64,sigma=.025):
    if not isinstance(arm,SearchArm) or type(seed) is not int or seed < 0 or type(generations) is not int or generations < 0 or population_size != 64 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('invalid P1 search settings')
    started = perf_counter()
    initial_rng = np.random.default_rng(np.random.SeedSequence([seed,0]))
    proposal = np.random.default_rng(np.random.SeedSequence([seed,1,list(SearchArm).index(arm)]))
    selection = np.random.default_rng(np.random.SeedSequence([seed,2,list(SearchArm).index(arm)]))
    values = initial_rng.uniform(.10,.15,64)
    ids = np.arange(64)
    low,high = float('inf'),float('-inf')
    low_id,high_id = -1,-1
    checks = 0
    first: int | str = 'NotReached'
    trajectory = []

    def archive(candidates,candidate_ids):
        nonlocal low,high,low_id,high_id,checks,first
        for value,identifier in zip(candidates,candidate_ids):
            # Verification independent of selection. Exact conversion preserves
            # float proposals; rational response equations then verify feasibility.
            model = ResponseModel(F(float(value)))
            if sum(model.weights) != 1 or min(model.weights) < 0 or model.marginals() != (F(1,4),F(3,4)):
                raise ArithmeticError('candidate verification failed')
            q = float(model.query())
            checks += 1
            if q < low:
                low,low_id = q,int(identifier)
            if q > high:
                high,high_id = q,int(identifier)
            if first == 'NotReached' and low <= .01 and high >= .99:
                first = checks
        trajectory.append(high-low)

    archive(values,ids)
    for generation in range(generations):
        new_ids = np.arange(64+generation*64,128+generation*64)
        if arm == SearchArm.UNIFORM:
            children = proposal.uniform(0,.25,64)
        elif arm == SearchArm.DIRECTED:
            children = np.concatenate([np.clip(pool[proposal.integers(0,32,32)] + proposal.normal(0,sigma,32),0,.25) for pool in (values[:32],values[32:])])
        else:
            children = np.clip(values[proposal.integers(0,64,64)] + proposal.normal(0,sigma,64),0,.25)
        archive(children,new_ids)
        if arm == SearchArm.UNIFORM:
            values,ids = children,new_ids
        elif arm == SearchArm.DIRECTED:
            pools, pool_ids = [],[]
            for side in (0,1):
                v = np.concatenate((values[side*32:(side+1)*32],children[side*32:(side+1)*32]))
                i = np.concatenate((ids[side*32:(side+1)*32],new_ids[side*32:(side+1)*32]))
                rank = np.lexsort((i,v if side == 0 else -v))[:32]
                pools.append(v[rank]); pool_ids.append(i[rank])
            values,ids = np.concatenate(pools),np.concatenate(pool_ids)
        else:
            union,union_ids = np.concatenate((values,children)),np.concatenate((ids,new_ids))
            selected = selection.choice(128,64,replace=False) if arm == SearchArm.FIT else _farthest(union,union_ids,64)
            values,ids = union[selected],union_ids[selected]
    witnesses = WitnessRange(low,high,(f'{arm.value}:{seed}:{low_id}',f'{arm.value}:{seed}:{high_id}'))
    return SearchResult(arm,seed,checks,checks,witnesses,Identification.NON_IDENTIFIED if low < high else Identification.NOT_ESTABLISHED,Scope.POPULATION,tuple(trajectory),first,(low/4,high/4),perf_counter()-started)


def marginal_conflict(declared: Interval, additional: Interval):
    """Exact empty intersection certificate for two constraints on P(Y0=1)."""
    return max(declared.lower,additional.lower) > min(declared.upper,additional.upper)
