from fractions import Fraction as F
import json
from pathlib import Path
import pytest
from bcs.simulator import Action, Variable as V, Clamp, Step, DeviceState, WorldSpec, Regime, advance_device, IndexedNoise, simulate

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / 'experiments/fixtures/causal-oracles-v0.1.json').read_text())

@pytest.mark.parametrize('regime,key', [(Regime.STOCHASTIC_PARTIAL,'independent_xor'), (Regime.CORRELATED_PARTIAL,'shared_xor'), (Regime.WRONG_GATE_PARTIAL,'wrong_gate')])
def test_interventional_laws_match_independent_fixtures(regime, key):
    spec = WorldSpec.for_regime(regime)
    for cell in FIXTURE['noise_diagnostics']['cells']:
        p = sum(weight for nm, ny, weight in spec.noise.support() if advance_device(DeviceState(0,0,0,0), Action.HOLD, ((V.R,cell['r']), (V.C,cell['c'])), nm, ny, spec.wrong_gate).y)
        assert p == F(str(cell[key]))


def test_clamps_replace_equations_and_only_c_r_persist():
    state = DeviceState(0,0,0,0)
    state = advance_device(state, Action.HOLD, ((V.M,1), (V.C,1)), 0,0,False)
    assert state == DeviceState(0,1,1,0)
    state = advance_device(state, Action.HOLD, (), 0,0,False)
    assert state == DeviceState(0,0,1,0)
    state = advance_device(state, Action.STOP, ((V.R,1), (V.C,0)), 0,0,False)
    assert state == DeviceState(1,1,0,1)


def test_shared_clock_advances_other_device_and_noise_is_indexed():
    noise = IndexedNoise(7)
    spec = WorldSpec.for_regime(Regime.CORRELATED_PARTIAL)
    before = noise.pair(spec.noise, 'ep',1,3)
    noise.pair(spec.noise, 'other',0,999)
    assert noise.pair(spec.noise, 'ep',1,3) == before
    assert before[0] == before[1]
    trace = simulate(spec, 'ep', 7, (Step.command(0,Action.START), Step.command(1,Action.START)))
    assert len(trace.states) == 3
    assert trace.states[-1][0].r == trace.states[-1][1].r == 1


def test_duplicate_clamps_and_nonbinary_values_are_rejected():
    with pytest.raises(ValueError):
        Step((Action.HOLD,Action.HOLD), (Clamp(0,V.R,0), Clamp(0,V.R,1)))
    with pytest.raises(ValueError):
        Clamp(0,V.R,2)
