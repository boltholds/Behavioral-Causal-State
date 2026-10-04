import json
import pytest
from bcs.generator import generate, compile_history, render, Observation, Executed, Hypothetical, Denied, Unknown, Forced, public_record, semantic_key
from bcs.simulator import Regime, WorldSpec, Variable as V, Action
from bcs.inference import evaluate, Predictive, ExactDistribution

@pytest.mark.parametrize('regime', list(Regime))
def test_generated_history_is_truthful_and_hides_unobserved_m(regime):
    spec = WorldSpec.for_regime(regime)
    for seed in range(6):
        generated = generate(spec, seed, f'ep-{seed}')
        assert len(generated.events) in (4,8,12,16)
        assert any(isinstance(e,Executed) and e.action != Action.HOLD for e in generated.events)
        assert [e.object_id for e in generated.events[:2]] == [0,1]
        history = compile_history(generated.events)
        for event in generated.events:
            if isinstance(event,Observation):
                assert event.variable in spec.visible
                assert event.value == generated.trace.states[event.time][event.object_id].get(event.variable)
        result = evaluate(spec,history,Predictive(0,(),(V.Y,)))
        assert isinstance(result,ExactDistribution)
        record = public_record(generated, include_labels=False)
        assert set(record) == {'group_id','messages'}
        assert all(set(m) == {'role','speaker_id','text'} for m in record['messages'])


def test_hypothetical_denied_and_unknown_do_not_advance_or_add_evidence():
    events = (Observation(0,0,V.C,0),Hypothetical(0,0,Action.START),Denied(0,0,Action.STOP),Unknown(0,0,V.Y))
    history = compile_history(events)
    assert len(history.steps) == 0
    assert len(history.evidence) == 1


def test_explicit_time_permutation_preserves_physical_history():
    events = (Observation(0,0,V.C,0),Executed(1,0,Action.START),Executed(2,0,Action.STOP))
    assert compile_history(events) == compile_history(tuple(reversed(events)))


def test_seed_repeatability_and_paraphrase_leave_semantics_unchanged():
    spec = WorldSpec.for_regime(Regime.STOCHASTIC_PARTIAL)
    a = generate(spec,12,'ep')
    b = generate(spec,12,'ep')
    assert a == b
    assert render(a.events,a.names,0) != render(a.events,a.names,1)
    assert semantic_key(a.events) == semantic_key(b.events)


def test_missing_executed_time_is_rejected():
    with pytest.raises(ValueError):
        compile_history((Executed(2,0,Action.START),))


def test_message_local_nonexecution_preserves_executed_transition():
    events = (Observation(0,0,V.C,0),Executed(1,0,Action.START),Denied(1,0,Action.START))
    history = compile_history(events)
    answer = evaluate(WorldSpec.for_regime(Regime.DETERMINISTIC),history,Predictive(0,(),(V.Y,)))
    assert answer.probability((1,)) == 1


def test_no_new_measurement_does_not_erase_previous_factual_evidence():
    events = (Observation(0,0,V.C,1),Unknown(0,0,V.C))
    answer = evaluate(WorldSpec.for_regime(Regime.DETERMINISTIC),compile_history(events),Predictive(0,(),(V.C,)))
    assert answer.probability((1,)) == 1
