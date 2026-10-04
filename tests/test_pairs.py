import pytest
from bcs.pairs import PairKind, HoldoutStratum, make_pair, verify_pair, rebuild_events
from bcs.generator import compile_history, Observation, Executed
from bcs.simulator import WorldSpec,Regime,Action,Variable as V

@pytest.mark.parametrize('kind',list(PairKind))
@pytest.mark.parametrize('stratum',list(HoldoutStratum))
def test_pair_is_truthful_and_has_checked_relation(kind,stratum):
    pair = make_pair(41,'pair-41',kind,stratum)
    assert 4 <= len(pair.left.events) == len(pair.right.events) <= 16
    assert verify_pair(pair)
    assert pair.left.messages != pair.right.messages
    assert all(any(isinstance(e,Executed) for e in side.events) for side in (pair.left,pair.right))
    if kind in (PairKind.PARAPHRASE,PairKind.TIME):
        assert compile_history(pair.left.events) == compile_history(pair.right.events)
    if kind == PairKind.ORDER:
        assert pair.annotation.tv == 1
        assert pair.explicit_time is False
    if stratum == HoldoutStratum.TEMPLATES:
        assert all(f >= 2 for f in pair.families)
    else:
        assert all(f < 2 for f in pair.families)


def test_rebuild_regenerates_measurements_after_semantic_change():
    spec=WorldSpec.for_regime(Regime.DETERMINISTIC)
    plan=(Observation(0,0,V.C,0),Observation(0,1,V.C,1),Executed(1,0,Action.START),Observation(1,0,V.R,0))
    events=rebuild_events(spec,0,'rebuild',plan)
    assert events[-1].value == 1


def test_same_seed_is_reproducible_without_model_outputs():
    assert make_pair(9,'same',PairKind.MODE,HoldoutStratum.NAMES) == make_pair(9,'same',PairKind.MODE,HoldoutStratum.NAMES)


def test_mislabeled_pair_kind_does_not_pass_relation_audit():
    from dataclasses import replace
    pair=make_pair(9,'wrong-kind',PairKind.ORDER,HoldoutStratum.NAMES)
    assert not verify_pair(replace(pair,kind=PairKind.BINDING))


def test_implicit_text_requires_times_derived_from_message_order():
    from dataclasses import replace
    pair=make_pair(9,'implicit-clock',PairKind.ORDER,HoldoutStratum.NAMES)
    # Swap only hidden event timestamps: the visible untimed text stays identical.
    events=pair.left.events
    wrong=events[:-2]+(replace(events[-2],time=events[-1].time),replace(events[-1],time=events[-2].time))
    mutated=replace(pair,left=replace(pair.left,events=wrong))
    from bcs.pairs import annotate
    mutated=replace(mutated,annotation=annotate(wrong,pair.right.events))
    assert not verify_pair(mutated)
