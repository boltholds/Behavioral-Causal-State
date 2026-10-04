from dataclasses import replace
from fractions import Fraction as F
import pytest
from bcs.calibration import calibration_panel, learn_core, BooleanCore, IncompleteCalibration, ConflictingCalibration
from bcs.simulator import WorldSpec, Regime, Variable as V, DeviceState, Action, Clamp, Step
from bcs.inference import Evidence, History, Interventional, evaluate


def test_twelve_visible_probes_learn_all_cells_and_count_preparation():
    panel = calibration_panel(WorldSpec.for_regime(Regime.DETERMINISTIC))
    assert len(panel.probes) == 12
    assert panel.action_count == 22
    assert all(len(step.clamps) <= 1 for p in panel.probes for step in p.history.steps)
    core = learn_core(panel.probes)
    assert isinstance(core,BooleanCore)
    assert core.r_table == (1,0,0,1,0,1)
    assert core.m_table == (0,1)
    assert core.y_table == (0,0,1,0)
    assert BooleanCore.from_wire(core.to_wire()) == core


def test_learning_changed_mechanism_cannot_copy_simulator_default_formula():
    spec = replace(WorldSpec.for_regime(Regime.DETERMINISTIC),wrong_gate=True)
    core = learn_core(calibration_panel(spec).probes)
    assert core.y_table == (0,0,0,1)
    answer = evaluate(core,History((),()),Interventional(0,(Clamp(0,V.R,1),Clamp(0,V.C,1)),(V.Y,)))
    assert answer.probability((1,)) == 1


def test_missing_cells_do_not_get_oracle_completion():
    probes = calibration_panel(WorldSpec.for_regime(Regime.DETERMINISTIC)).probes
    result = learn_core(probes[:-1])
    assert isinstance(result,IncompleteCalibration)
    assert len(result.missing_cells) == 1


def test_conflicting_observation_stops_freezing():
    probes = calibration_panel(WorldSpec.for_regime(Regime.DETERMINISTIC)).probes
    p = probes[-1]
    evidence = tuple(replace(e,value=1-e.value) if e.variable == V.Y else e for e in p.history.evidence)
    result = learn_core(probes+(replace(p,history=replace(p.history,evidence=evidence)),))
    assert isinstance(result,ConflictingCalibration)


def test_clamped_child_does_not_teach_natural_mechanism():
    probes = calibration_panel(WorldSpec.for_regime(Regime.DETERMINISTIC)).probes
    p = probes[-1]
    steps = list(p.history.steps)
    last = steps[-1]
    steps[-1] = replace(last,clamps=last.clamps+(Clamp(0,V.Y,0),))
    with pytest.raises(ValueError,match='clamped child'):
        learn_core(probes[:-1]+(replace(p,history=replace(p.history,steps=tuple(steps))),))
