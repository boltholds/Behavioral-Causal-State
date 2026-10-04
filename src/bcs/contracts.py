"""Immutable answer variants. Certificates refer to independently checkable artifacts.

Construction validates status consistency; it does not establish the truth of a proof.
Only a verifier may attach a certificate after checking its referenced artifact.
"""
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite


class Identification(StrEnum):
    IDENTIFIED = 'ProvenIdentified'
    NON_IDENTIFIED = 'ProvenNonIdentified'
    NOT_ESTABLISHED = 'NotEstablished'
    NA = 'NotApplicable'

class Scope(StrEnum):
    POPULATION = 'PopulationLaw'
    FINITE_DATA = 'FiniteDataConstraints'
    FINITE_CLASS = 'DeclaredFiniteClass'

class Computation(StrEnum):
    COMPLETE = 'Complete'
    BUDGET = 'BudgetExhausted'
    NUMERICAL_FAILURE = 'NumericalFailure'
    NOT_STARTED = 'NotStarted'

class Feasibility(StrEnum):
    WITNESSES = 'FeasibleWitnesses'
    INFEASIBLE = 'CertifiedInfeasible'
    NOT_ESTABLISHED = 'NotEstablished'

class BoundsPrecision(StrEnum):
    SHARP = 'Sharp'
    TOLERANCE = 'CertifiedWithinTolerance'
    OUTER = 'OuterOnly'
    NA = 'NotApplicable'

class Grounding(StrEnum):
    RESOLVED = 'Resolved'
    AMBIGUOUS = 'Ambiguous'
    INVALID = 'Invalid'


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError('expected finite probability in [0,1]')

@dataclass(frozen=True)
class Point:
    value: float
    def __post_init__(self):
        probability(self.value)

@dataclass(frozen=True)
class StatisticalEstimate:
    value: float
    lower: float
    upper: float
    confidence: float
    def __post_init__(self):
        for value in (self.value, self.lower, self.upper, self.confidence):
            probability(value)
        if not self.lower <= self.value <= self.upper or not 0 < self.confidence < 1:
            raise ValueError('invalid statistical interval')

@dataclass(frozen=True)
class CertifiedBounds:
    lower: float
    upper: float
    def __post_init__(self):
        probability(self.lower)
        probability(self.upper)
        if self.lower > self.upper:
            raise ValueError('reversed bounds')

@dataclass(frozen=True)
class WitnessRange:
    lower: float
    upper: float
    model_ids: tuple[str, ...]
    def __post_init__(self):
        CertifiedBounds(self.lower, self.upper)
        if not isinstance(self.model_ids, tuple) or not self.model_ids:
            raise ValueError('witnesses required')
        if self.lower != self.upper and len(set(self.model_ids)) < 2:
            raise ValueError('distinct extremal witnesses required')

@dataclass(frozen=True)
class NoAnswer:
    reason: str

@dataclass(frozen=True)
class Undefined:
    reason: str

@dataclass(frozen=True)
class Certificate:
    method: str
    artifact_ids: tuple[str, ...]
    def __post_init__(self):
        if not self.method or not isinstance(self.artifact_ids, tuple) or not self.artifact_ids:
            raise ValueError('proof artifact references required')

@dataclass(frozen=True)
class NoCertificate:
    pass

Result = Point | StatisticalEstimate | CertifiedBounds | WitnessRange | NoAnswer | Undefined

@dataclass(frozen=True)
class AnswerRecord:
    snapshot_id: str
    history_id: str
    query_id: str
    model_class_id: str
    assumption_ids: tuple[str, ...]
    tolerance: float
    identification: Identification
    scope: Scope
    computation: Computation
    feasibility: Feasibility
    bounds_precision: BoundsPrecision
    grounding: Grounding
    result: Result
    evidence: Certificate | NoCertificate

    def __post_init__(self):
        for name in ('snapshot_id', 'history_id', 'query_id', 'model_class_id'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} required')
        if not isinstance(self.assumption_ids, tuple) or not all(isinstance(x, str) and x for x in self.assumption_ids):
            raise ValueError('immutable assumption IDs required')
        if not isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError('invalid tolerance')
        for name, kind in ENUM_FIELDS.items():
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f'{name} must be {kind.__name__}')
        if type(self.result) not in RESULT_TYPES.values() or type(self.evidence) not in (Certificate, NoCertificate):
            raise ValueError('invalid result/evidence variant')
        proven = self.identification in (Identification.IDENTIFIED, Identification.NON_IDENTIFIED)
        if (proven or self.feasibility == Feasibility.INFEASIBLE or isinstance(self.result, CertifiedBounds)) and not isinstance(self.evidence, Certificate):
            raise ValueError('claim requires certificate')
        if self.identification == Identification.IDENTIFIED and isinstance(self.result, WitnessRange):
            raise ValueError('search unanimity is not identification')
        if self.feasibility == Feasibility.INFEASIBLE:
            if proven or not isinstance(self.result, (Undefined, NoAnswer)):
                raise ValueError('empty class has no identified answer')
        if self.bounds_precision != BoundsPrecision.NA and not isinstance(self.result, CertifiedBounds):
            raise ValueError('only certified bounds have a bounds precision')
        if isinstance(self.result, CertifiedBounds) and self.bounds_precision == BoundsPrecision.NA:
            raise ValueError('bounds precision required')
        if self.grounding == Grounding.INVALID and not isinstance(self.result, (Undefined, NoAnswer)):
            raise ValueError('invalid grounding cannot yield a probability')

    def to_wire(self):
        wire = asdict(self)
        for name in ENUM_FIELDS:
            wire[name] = getattr(self, name).value
        wire['result']['kind'] = type(self.result).__name__
        wire['evidence']['kind'] = type(self.evidence).__name__
        return wire

ENUM_FIELDS = dict(identification=Identification, scope=Scope, computation=Computation,
                  feasibility=Feasibility, bounds_precision=BoundsPrecision, grounding=Grounding)
RESULT_TYPES = {c.__name__: c for c in (Point, StatisticalEstimate, CertifiedBounds, WitnessRange, NoAnswer, Undefined)}


def answer_from_wire(wire):
    fields = dict(wire)
    for name, kind in ENUM_FIELDS.items():
        fields[name] = kind(fields[name])
    fields['assumption_ids'] = tuple(fields['assumption_ids'])
    result = dict(fields['result'])
    kind = RESULT_TYPES[result.pop('kind')]
    if kind is WitnessRange:
        result['model_ids'] = tuple(result['model_ids'])
    fields['result'] = kind(**result)
    proof = dict(fields['evidence'])
    proof_kind = proof.pop('kind')
    if proof_kind == 'Certificate':
        proof['artifact_ids'] = tuple(proof['artifact_ids'])
        fields['evidence'] = Certificate(**proof)
    elif proof_kind == 'NoCertificate' and not proof:
        fields['evidence'] = NoCertificate()
    else:
        raise ValueError('unknown certificate variant')
    return AnswerRecord(**fields)
