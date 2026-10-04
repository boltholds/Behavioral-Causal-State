from fractions import Fraction as F
import pytest
from bcs.simulator import Action, Variable as V, Clamp, Step, WorldSpec, Regime
from bcs.inference import Evidence, History, Predictive, Interventional, Counterfactual, ExactDistribution, InferenceUndefined, InferenceIncomplete, evaluate


def test_factual_evidence_abducts_shared_noise_for_past_counterfactual():
    spec = WorldSpec.for_regime(Regime.STOCHASTIC_PARTIAL)
    history = History((Step.command(0,Action.START),), (Evidence(0,0,V.C,0),Evidence(1,0,V.Y,1)))
    cf = evaluate(spec, history, Counterfactual(0,1,Action.STOP,(V.Y,)))
    assert isinstance(cf, ExactDistribution)
    # Y_f=1 means NM xor NY=0. Replacing start with stop flips Y under same U.
    assert cf.probability((0,)) == 1
    fresh = evaluate(spec, history, Interventional(0,(Clamp(0,V.R,0),), (V.Y,)))
    assert fresh.probability((1,)) == F(18,100)
    assert cf.evidence_probability == F(41,100)  # initial C=0 has mass .5


def test_later_measurements_are_evidence_not_cf_assignments():
    spec = WorldSpec.for_regime(Regime.DETERMINISTIC)
    h = History((Step.command(0,Action.START),Step.hold()), (Evidence(0,0,V.C,0),Evidence(2,0,V.Y,1)))
    result = evaluate(spec,h,Counterfactual(0,1,Action.STOP,(V.M,V.Y)))
    assert result.probability((0,0)) == 1


def test_cf_keeps_later_factual_actions_and_clamps():
    h = History((Step.command(0,Action.START),Step.command(0,Action.START)), ())
    result = evaluate(WorldSpec.for_regime(Regime.DETERMINISTIC),h,Counterfactual(0,1,Action.STOP,(V.R,)))
    assert result.probability((1,)) == 1


def test_zero_support_on_other_object_is_undefined():
    h = History((),(Evidence(0,1,V.R,1),))
    result = evaluate(WorldSpec.for_regime(Regime.DETERMINISTIC),h,Predictive(0,(Step.hold(),),(V.Y,)))
    assert isinstance(result, InferenceUndefined)


def test_future_noise_integrated_and_budget_not_point_answer():
    h = History((),(Evidence(0,0,V.C,1),))
    spec = WorldSpec.for_regime(Regime.STOCHASTIC_PARTIAL)
    result = evaluate(spec,h,Predictive(0,(Step.hold(),),(V.Y,)))
    assert result.probability((1,)) == F(1,10)
    assert isinstance(evaluate(spec,h,Predictive(0,(Step.hold(),),(V.Y,)),budget=0),InferenceIncomplete)


def test_invalid_temporal_reference_is_rejected():
    with pytest.raises(ValueError):
        History((),(Evidence(1,0,V.R,0),))
    with pytest.raises(ValueError):
        evaluate(WorldSpec.for_regime(Regime.DETERMINISTIC),History((Step.hold(),),()),Counterfactual(0,1,Action.START,(V.Y,)))
