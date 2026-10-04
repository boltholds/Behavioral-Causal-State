import pytest
from bcs.contracts import AnswerRecord, Identification, Computation, Scope, Feasibility, BoundsPrecision, Grounding, Point, WitnessRange, NoAnswer, Certificate, NoCertificate, answer_from_wire


def valid_answer(**overrides):
    fields = dict(snapshot_id='fixture-snapshot-1', history_id='h1', query_id='q1', model_class_id='singleton',
                  assumption_ids=('known_scm',), tolerance=0.0,
                  identification=Identification.IDENTIFIED, scope=Scope.FINITE_CLASS,
                  computation=Computation.COMPLETE, feasibility=Feasibility.WITNESSES,
                  bounds_precision=BoundsPrecision.NA, grounding=Grounding.RESOLVED,
                  result=Point(0.5), evidence=Certificate('exhaustive', ('model-1',)))
    return AnswerRecord(**(fields | overrides))


def test_answer_roundtrip_preserves_scopes_and_proof():
    answer = valid_answer()
    assert answer_from_wire(answer.to_wire()) == answer


def test_unanimous_search_is_not_identification():
    with pytest.raises(ValueError):
        valid_answer(result=WitnessRange(0.5, 0.5, ('m1',)), evidence=NoCertificate())


def test_timeout_cannot_claim_complete_identification_without_certificate():
    with pytest.raises(ValueError):
        valid_answer(computation=Computation.BUDGET, evidence=NoCertificate())


def test_empty_search_cannot_claim_infeasible():
    with pytest.raises(ValueError):
        valid_answer(identification=Identification.NOT_ESTABLISHED, feasibility=Feasibility.INFEASIBLE,
                     result=NoAnswer('no candidates'), evidence=NoCertificate())
