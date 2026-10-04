from fractions import Fraction as F
from dataclasses import replace
import pytest
from bcs.misspecification import Interval, NoiseClass, Compatible, Incompatible, NotEstablished, certify, verify, clopper_pearson, population_panel, run_panel

@pytest.mark.parametrize('regime,expected', [('independent_xor',(True,True)),('shared_xor',(False,True)),('wrong_gate',(False,False))])
def test_exact_compatibility_matrix_and_independent_certificate_verification(regime,expected):
    panel = tuple(Interval(x,x) for x in population_panel(regime))
    for model,want in zip(NoiseClass,expected):
        result = certify(panel,model)
        assert isinstance(result,Compatible) == want
        assert verify(panel,model,result)


def test_corrupt_witness_is_rejected():
    panel = tuple(Interval(x,x) for x in population_panel('independent_xor'))
    result = certify(panel,NoiseClass.INDEPENDENT)
    assert not verify(panel,NoiseClass.INDEPENDENT,replace(result,weights=(F(1),F(0),F(0),F(0))))


def test_full_intervals_admit_boundary_and_half_noise():
    panel = (Interval(F(0),F(1)),)*4
    assert isinstance(certify(panel,NoiseClass.INDEPENDENT),Compatible)
    half = (Interval(F(1,2),F(1,2)),)*4
    assert verify(half,NoiseClass.INDEPENDENT,certify(half,NoiseClass.INDEPENDENT))


def test_exhausted_budget_never_claims_infeasible():
    panel = tuple(Interval(x,x) for x in population_panel('wrong_gate'))
    result = certify(panel,NoiseClass.INDEPENDENT,budget=0)
    assert isinstance(result,NotEstablished)
    assert result.reason == 'BudgetExhausted'
    assert not verify(panel,NoiseClass.INDEPENDENT,result)


def test_cp_intervals_keep_zero_and_one_and_expand_outward():
    zero = clopper_pearson(0,100,.05)
    full = clopper_pearson(100,100,.05)
    assert zero.lower == 0 and 0.036 < float(zero.upper) < .037
    assert full.upper == 1 and .963 < float(full.lower) < .964
    with pytest.raises(ValueError):
        clopper_pearson(101,100,.05)


def test_finite_panel_counts_data_not_hidden_m():
    report = run_panel(seeds=(0,1),episodes=10000)
    assert report['episodes'] == 240000
    for regime in ('shared_xor','wrong_gate'):
        assert report['rejections'][regime]['H_ind'] == 2
    assert report['rejections']['shared_xor']['H_joint'] == 0


def test_cp_endpoint_encloses_high_precision_reference_quantile():
    # Independent 65-digit inverse-beta reference; alpha/2 = 1/480.
    from decimal import Decimal
    interval = clopper_pearson(1000,10000,.05/12)
    reference = Decimal('0.09158857593751597758728772363287186247201683692797816926708595485')
    assert F(interval.lower) <= F(reference)


def test_directed_binomial_sum_encloses_exact_small_case():
    from bcs.misspecification import binomial_cdf_enclosure
    lo,hi = binomial_cdf_enclosure(1,3,F(1,3))
    assert F(lo) <= F(20,27) <= F(hi)
    assert hi-lo < F(1,10**40)


def test_numerical_failure_does_not_become_infeasible(monkeypatch):
    import bcs.misspecification as misspec
    def cannot_certify(*args):
        raise ArithmeticError('endpoint not verified')
    monkeypatch.setattr(misspec,'clopper_pearson',cannot_certify)
    result = misspec.run_panel(seeds=(0,),episodes=10)
    assert all(s == 'NotEstablished:NumericalFailure' for row in result['rows'] for s in row['statuses'].values())
