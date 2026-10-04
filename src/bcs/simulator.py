"""SPEC-04 binary device dynamics. PhysicalTrace is evaluator-only."""
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction as F
from hashlib import sha256
import json

class Action(StrEnum):
    START = 'start'
    STOP = 'stop'
    HOLD = 'hold'

class Variable(StrEnum):
    R = 'R'
    M = 'M'
    C = 'C'
    Y = 'Y'

class Regime(StrEnum):
    DETERMINISTIC = 'deterministic'
    STOCHASTIC_FULL = 'stochastic_full'
    STOCHASTIC_PARTIAL = 'stochastic_partial'
    CORRELATED_PARTIAL = 'correlated_partial'
    WRONG_GATE_PARTIAL = 'wrong_gate_partial'


def bit(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError('binary integer required')


def object_id(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError('object must be 0 or 1')

@dataclass(frozen=True)
class DeviceState:
    r: int
    m: int
    c: int
    y: int
    def __post_init__(self):
        for value in (self.r,self.m,self.c,self.y):
            bit(value)
    def get(self, variable: Variable):
        if not isinstance(variable, Variable):
            raise ValueError('Variable enum required')
        return getattr(self,variable.value.lower())

@dataclass(frozen=True)
class Clamp:
    object_id: int
    variable: Variable
    value: int
    def __post_init__(self):
        object_id(self.object_id)
        bit(self.value)
        if not isinstance(self.variable, Variable):
            raise ValueError('Variable enum required')

@dataclass(frozen=True)
class Step:
    actions: tuple[Action, Action]
    clamps: tuple[Clamp, ...] = ()
    def __post_init__(self):
        if not isinstance(self.actions,tuple) or len(self.actions) != 2 or not all(isinstance(a,Action) for a in self.actions):
            raise ValueError('two typed actions required')
        if not isinstance(self.clamps,tuple) or not all(isinstance(c,Clamp) for c in self.clamps):
            raise ValueError('immutable clamps required')
        if len({(c.object_id,c.variable) for c in self.clamps}) != len(self.clamps):
            raise ValueError('duplicate clamp target')
    @classmethod
    def hold(cls):
        return cls((Action.HOLD,Action.HOLD))
    @classmethod
    def command(cls, obj, action):
        object_id(obj)
        return cls((action,Action.HOLD) if obj == 0 else (Action.HOLD,action))
    def local_clamps(self,obj):
        return tuple((c.variable,c.value) for c in self.clamps if c.object_id == obj)

@dataclass(frozen=True)
class NoiseLaw:
    """Exact joint probabilities in order NM,NY = 00,01,10,11."""
    weights: tuple[F,F,F,F]
    def __post_init__(self):
        if not isinstance(self.weights,tuple) or len(self.weights) != 4 or not all(isinstance(x,F) and x >= 0 for x in self.weights) or sum(self.weights) != 1:
            raise ValueError('normalized exact joint noise law required')
    def support(self):
        return tuple((i//2,i%2,p) for i,p in enumerate(self.weights) if p)
    @classmethod
    def independent(cls,a=F(1,10),b=F(1,10)):
        if not 0 <= a <= 1 or not 0 <= b <= 1:
            raise ValueError('invalid noise probability')
        return cls(tuple(F(p) for p in ((1-a)*(1-b),(1-a)*b,a*(1-b),a*b)))

@dataclass(frozen=True)
class WorldSpec:
    regime: Regime
    noise: NoiseLaw
    wrong_gate: bool = False
    @classmethod
    def for_regime(cls,regime):
        if not isinstance(regime,Regime):
            raise ValueError('Regime enum required')
        if regime == Regime.DETERMINISTIC:
            noise = NoiseLaw((F(1),F(0),F(0),F(0)))
        elif regime == Regime.CORRELATED_PARTIAL:
            noise = NoiseLaw((F(9,10),F(0),F(0),F(1,10)))
        else:
            noise = NoiseLaw.independent()
        return cls(regime,noise,regime == Regime.WRONG_GATE_PARTIAL)
    @property
    def visible(self):
        if self.regime in (Regime.DETERMINISTIC,Regime.STOCHASTIC_FULL):
            return tuple(Variable)
        return (Variable.R,Variable.C,Variable.Y)


def advance_device(previous,action,clamps,nm,ny,wrong_gate):
    """Hard replacement before descendants; C/R persist, M/Y recompute."""
    if not isinstance(action,Action):
        raise ValueError('Action enum required')
    bit(nm)
    bit(ny)
    values = dict(clamps)
    if len(values) != len(clamps):
        raise ValueError('duplicate local clamps')
    for variable,value in clamps:
        if not isinstance(variable,Variable):
            raise ValueError('Variable enum required')
        bit(value)
    c = values.get(Variable.C,previous.c)
    r = values.get(Variable.R,1 if action == Action.START else 0 if action == Action.STOP else previous.r)
    m = values.get(Variable.M,r ^ nm)
    y = values.get(Variable.Y,(m & (c if wrong_gate else 1-c)) ^ ny)
    return DeviceState(r,m,c,y)

@dataclass(frozen=True)
class IndexedNoise:
    seed: int
    def uniform(self,episode,obj,channel,time):
        encoded = json.dumps([self.seed,episode,obj,channel,time],separators=(',',':'),ensure_ascii=False).encode()
        return F(int.from_bytes(sha256(encoded).digest(),'big'),2**256)
    def initial_c(self,episode,obj):
        return int(self.uniform(episode,obj,'initial-C',0) < F(1,2))
    def pair(self,law,episode,obj,time):
        # One indexed joint driver represents the within-slice noise vector.
        # Re-evaluation order and intervention path do not consume an RNG stream.
        u = self.uniform(episode,obj,'joint-NM-NY',time)
        total = F(0)
        for nm,ny,p in law.support():
            total += p
            if u < total:
                return nm,ny
        raise ArithmeticError('noise law not normalized')

@dataclass(frozen=True)
class PhysicalTrace:
    episode_id: str
    states: tuple[tuple[DeviceState,DeviceState], ...]
    steps: tuple[Step, ...]
    noise_pairs: tuple[tuple[tuple[int,int],tuple[int,int]], ...]


def simulate(spec,episode_id,seed,steps):
    if not isinstance(steps,tuple) or not all(isinstance(s,Step) for s in steps):
        raise ValueError('immutable steps required')
    noise = IndexedNoise(seed)
    previous = tuple(DeviceState(0,0,noise.initial_c(episode_id,obj),0) for obj in range(2))
    states, pairs = [], []
    for time,step in enumerate((Step.hold(),) + steps):
        pair = tuple(noise.pair(spec.noise,episode_id,obj,time) for obj in range(2))
        previous = tuple(advance_device(previous[obj],step.actions[obj],step.local_clamps(obj),*pair[obj],spec.wrong_gate) for obj in range(2))
        states.append(previous)
        pairs.append(pair)
    return PhysicalTrace(episode_id,tuple(states),steps,tuple(pairs))
