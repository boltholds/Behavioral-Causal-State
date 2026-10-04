from fractions import Fraction as F
import pytest
from bcs.population import ResponseModel, response_bounds, search, SearchArm
from bcs.contracts import CertifiedBounds, Undefined, Scope, WitnessRange, Identification
from bcs.misspecification import Interval


def test_same_do_laws_different_counterfactuals():
    minus,plus = ResponseModel(F(0)),ResponseModel(F(1,4))
    assert minus.marginals() == plus.marginals() == (F(1,4),F(3,4))
    assert minus.query() == 0 and plus.query() == 1
    assert response_bounds(Interval(F(1,4),F(1,4)),Interval(F(3,4),F(3,4))) == CertifiedBounds(0,1)
    assert response_bounds(Interval(F(1,4),F(1,4)),Interval(F(3,4),F(3,4)),monotone=True) == CertifiedBounds(1,1)


def test_zero_denominator_requires_domain_policy():
    assert isinstance(response_bounds(Interval(F(0),F(1,4)),Interval(F(1,2),F(3,4))),Undefined)


def test_finite_marginal_intervals_have_distinct_sampling_uncertainty():
    assert response_bounds(Interval(F(1,5),F(3,10)),Interval(F(7,10),F(4,5))) == CertifiedBounds(0,1)

@pytest.mark.parametrize('arm',list(SearchArm))
def test_all_search_arms_count_every_candidate_and_keep_witnesses(arm):
    result = search(arm,11,generations=3)
    assert result.evaluations == 256
    assert result.feasibility_checks == result.evaluations
    assert isinstance(result.result,WitnessRange)
    assert result.identification != Identification.IDENTIFIED
    assert len(result.trajectory) == 4
    assert all(a <= b for a,b in zip(result.trajectory,result.trajectory[1:]))


def test_search_and_exact_bounds_remain_distinct():
    result = search(SearchArm.UNIFORM,2,generations=0)
    assert .39 < result.result.lower < result.result.upper < .61
    assert result.identification == Identification.NON_IDENTIFIED
    assert result.scope == Scope.POPULATION


def test_extra_marginal_constraint_requires_actual_empty_intersection_proof():
    from bcs.population import marginal_conflict
    assert marginal_conflict(Interval(F(1,4),F(1,4)),Interval(F(0),F(0)))
    assert not marginal_conflict(Interval(F(1,4),F(1,4)),Interval(F(0),F(1,2)))


def test_certified_bounds_contain_non_dyadic_exact_endpoint():
    bounds = response_bounds(Interval(F(3,4),F(3,4)),Interval(F(1,4),F(1,4)))
    assert F(bounds.upper) >= F(1,3)


def test_mandatory_controls_block_acceptance_even_with_successful_search():
    from bcs.cli import assess_p1
    summary = {arm.value:{'successful_seeds':100,'total_seeds':100,'evaluations_per_run':6464} for arm in SearchArm}
    controls = {key:{'passed':True} for key in ('coupling','monotonicity','finite_data','zero_denominator','infeasible','restricted_coupling')}
    assert assess_p1(summary,controls,95,100,6464)
    controls['monotonicity']['passed'] = False
    assert not assess_p1(summary,controls,95,100,6464)
