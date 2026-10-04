import pytest
from bcs.generator import generate, Observation, Unknown, Executed, Forced, Hypothetical, Denied
from bcs.simulator import WorldSpec, Regime, Variable as V, Action as A
from bcs.language_tokens import encode_events, decode_events, allowed_tokens, DecodedEvents, InvalidDecoding, EOS, PAD


def test_all_modes_round_trip_and_unknown_differs_from_false():
    events=(Observation(0,0,V.C,0),Observation(0,1,V.C,1),Executed(1,0,A.START),
            Forced(2,1,V.R,0),Unknown(2,0,V.Y),Hypothetical(2,0,A.STOP),Denied(2,1,A.START))
    tokens=encode_events(events)
    assert decode_events(tokens)==DecodedEvents(events)
    assert tokens[-1]==EOS
    for index,token in enumerate(tokens):
        assert token in allowed_tokens(tokens[:index])
    changed=events[:4]+(Observation(2,0,V.Y,0),)+events[5:]
    assert encode_events(changed)!=tokens


@pytest.mark.parametrize('length',[4,8,12,16])
def test_generated_history_round_trip(length):
    events=generate(WorldSpec.for_regime(Regime.DETERMINISTIC),9,'tokens',length).events
    assert decode_events(encode_events(events))==DecodedEvents(events)


def test_missing_eos_extra_token_and_impossible_clock_are_invalid():
    tokens=encode_events((Executed(1,0,A.START),))
    assert isinstance(decode_events(tokens[:-1]),InvalidDecoding)
    assert isinstance(decode_events(tokens+(PAD,)),InvalidDecoding)
    assert isinstance(decode_events(tokens[:-1]+tokens),InvalidDecoding)  # two transitions at t=1
    assert isinstance(decode_events((EOS,)),InvalidDecoding)


def test_type_masks_use_only_own_prefix_and_stop_after_eos():
    a=encode_events((Executed(1,0,A.START),));b=encode_events((Executed(1,0,A.STOP),))
    assert a[:5]==b[:5]
    assert a[5] in allowed_tokens(a[:5]) and b[5] in allowed_tokens(a[:5])
    assert allowed_tokens(a)==()
    with pytest.raises(ValueError):
        allowed_tokens((PAD,))
