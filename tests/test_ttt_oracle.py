"""Catch omitted reset, hidden fields, stochastic acceptance, and false EQ passes."""
import copy
import importlib.util
import pytest
from bcs.simulator import Regime


def api():
    from experiments.ttt import runner
    return runner


def test_ttt_oracle_is_available():
    assert importlib.util.find_spec('experiments.ttt.runner') is not None


def test_membership_outputs_full_after_step_and_resets():
    r = api()
    f = r.load_fixtures()[0]
    oracle = r.SimulatorOracle(f)
    # commands profile begins hold, start0, stop0, start1, stop1; C reset 00.
    assert oracle.membership([1, 3], 0) == [208, 221]
    assert oracle.membership([0], 0) == [0]
    assert oracle.membership([1, 3], 1) == [221]
    assert oracle.mq_count == 3
    assert oracle.symbols_executed == 5
    assert oracle.reset_count == 3


def test_stochastic_world_cannot_enter_deterministic_ttt():
    r = api()
    with pytest.raises(ValueError, match='deterministic'):
        r.SimulatorOracle(r.load_fixtures()[0], regime=Regime.STOCHASTIC_FULL)


def test_reference_minimum_and_exact_false_conjecture_detection():
    r = api()
    ref = r.build_reference(r.load_fixtures()[0])
    assert len(ref['transitions']) == 4
    assert r.minimal_state_count(ref) == 4
    trivial = {'initial': 0, 'transitions': [[[0, 0]] * 5]}
    assert r.product_counterexample(ref, trivial) == [1]
    corrupted = copy.deepcopy(ref)
    corrupted['transitions'][0][0][1] = 1
    assert r.product_counterexample(ref, corrupted) == [0]
    assert r.product_counterexample(ref, ref) is None


def test_partition_refinement_merges_transient_full_output_states():
    r = api()
    # State 1 has a different historical output but identical future outputs to 0.
    machine = {'initial': 0, 'transitions': [[[1, 3]], [[1, 3]]]}
    assert r.minimal_state_count(machine) == 1


def test_exact_equivalence_exhausts_product_edges():
    r = api()
    ref = r.build_reference(r.load_fixtures()[0])
    stats = {}
    assert r.product_counterexample(ref, ref, stats=stats) is None
    assert stats == {'pairs_visited': 4, 'edges_compared': 20}


@pytest.mark.parametrize('field,value', [('initial', -1), ('initial', True), ('destination', -1),
    ('destination', True), ('destination', 4), ('output', -1), ('output', True), ('output', 256)])
def test_product_eq_rejects_invalid_machine_indices_and_observations(field, value):
    r = api()
    ref = r.build_reference(r.load_fixtures()[0])
    damaged = copy.deepcopy(ref)
    if field == 'initial':
        damaged['initial'] = value
    else:
        damaged['transitions'][0][0][0 if field == 'destination' else 1] = value
    with pytest.raises(ValueError):
        r.product_counterexample(ref, damaged)
