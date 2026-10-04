"""Hand-derived L1 failures; generated smoke data never establishes full gates."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bcs.calibration import calibration_panel, learn_core
from bcs.dataset import DatasetCounts, build_dataset
from bcs.dataset_codec import event_from_wire
from bcs.generator import Observation, Unknown, Executed, Hypothetical, Denied
from bcs.language_tokens import encode_events, EOS
from bcs.simulator import Action as A, Variable as V, WorldSpec, Regime


def evaluator():
    from experiments.l1 import evaluation
    return evaluation


def core():
    return learn_core(calibration_panel(WorldSpec.for_regime(Regime.DETERMINISTIC)).probes)


def simple():
    return (Observation(0, 0, V.C, 1), Observation(0, 1, V.C, 0),
            Executed(1, 0, A.START), Executed(2, 0, A.STOP))


def test_invalid_output_penalizes_every_category_and_joint():
    result = evaluator().score_history((EOS,), simple(), core())
    for comparison in ('branch_to_oracle', 'branch_to_simulator'):
        assert result[comparison]['categories'] == [1, 1, 1, 1]
        assert result[comparison]['language'] == result[comparison]['joint'] == 1
    assert result['oracle_to_simulator']['language'] == 0
    assert result['coverage']['point_fraction'] == 0


def test_category_weights_are_history_then_category_not_query_pool():
    # START instead of the last STOP changes predictive M from 0 to 1, while
    # C interventions retain R and change M with the command. One CF reference is invalid
    # because the factual replacement no longer opposes the decoded action.
    events = simple()
    predicted = events[:-1] + (Executed(2, 0, A.START),)
    result = evaluator().score_history(encode_events(predicted), events, core())
    assert result['branch_to_oracle']['categories'] == [.5, .5, .5, 1]
    assert result['branch_to_oracle']['language'] == .625
    assert result['branch_to_oracle']['joint'] == 0


def test_knownness_zero_and_unknown_are_distinct_structural_fields():
    gold = simple() + (Observation(2, 0, V.R, 0),)
    predicted = simple() + (Unknown(2, 0, V.R),)
    report = evaluator().structural_scores([encode_events(predicted)], [gold], [False])
    assert report['full_history_error'] == 1
    assert report['fields']['mode']['errors'] == 1
    assert report['fields']['value']['errors'] == 1
    assert report['fields']['entity']['errors'] == 0
    assert report['critical_errors'] == [1]


def test_time_invariance_uses_timestamps_and_normalizes_narrative_order():
    events = simple()
    predicted = events[:-2] + tuple(reversed(events[-2:]))
    report = evaluator().structural_scores([encode_events(predicted)], [events], [True])
    assert report['full_history_error'] == report['full_event_error'] == 0
    assert report['critical_errors'] == [0]
    # Without explicit-time equivalence this is an order error.
    raw = evaluator().structural_scores([encode_events(predicted)], [events], [False])
    assert raw['full_history_error'] == 1


@pytest.mark.parametrize('replacement', [Hypothetical(1, 0, A.START), Denied(1, 0, A.START), Executed(1, 1, A.START)])
def test_critical_checks_mode_negation_and_binding_dependencies(replacement):
    gold = (Executed(1, 0, A.START),)
    # Hypothetical/denied t=0 are valid non-executed histories and must still
    # fail interpretation, rather than succeed through value-only comparison.
    if not isinstance(replacement, Executed):
        replacement = type(replacement)(0, replacement.object_id, replacement.action)
    report = evaluator().structural_scores([encode_events((replacement,))], [gold], [False])
    assert report['critical_errors'] == [1]


def test_exact_binomial_family_bound_and_bernstein_sample_ranges():
    e = evaluator()
    assert e.clopper_pearson_upper(0, 1000, .05 / 14) == pytest.approx(1 - (.05 / 14) ** (1 / 1000))
    assert e.clopper_pearson_upper(1000, 1000, .05 / 14) == 1
    assert e.paired_excess({'a': 0, 'b': 0}, {'b': 0, 'a': 0})['upper'] == 1
    result = e.paired_excess(dict.fromkeys(map(str, range(4000)), 0), dict.fromkeys(map(str, range(4000)), 0))
    assert result['upper'] == pytest.approx(14 * math.log(40) / (3 * 3999))
    assert result['upper'] < .005
    from bcs.metrics import empirical_bernstein_upper
    tv_upper = empirical_bernstein_upper([0] * 1000, 0, 1)
    assert tv_upper == pytest.approx(7 * math.log(40) / (3 * 999))
    assert tv_upper < .01
    with pytest.raises(ValueError, match='group'):
        e.paired_excess({'a': 0, 'b': 0}, {'b': 0, 'c': 0})


@pytest.fixture
def smoke(tmp_path):
    dataset = tmp_path / 'dataset'
    build_dataset(dataset, DatasetCounts(2, 2, 2, 1), seed=108)
    rows = [json.loads(line) for name in ('iid_test', 'challenge')
            for line in (dataset / f'evaluator/{name}.jsonl').read_text().splitlines()]
    gold = {side['record_id']: encode_events(tuple(event_from_wire(event) for event in side['events']))
            for row in rows for side in row['sides']}
    return dataset, rows, {name: dict(gold) for name in ('concat', 'attention', 'gru')}


def test_smoke_oracle_predictions_publish_groups_and_cannot_pass_full_l1(smoke, tmp_path):
    dataset, rows, predictions = smoke
    result = evaluator().evaluate_seed(dataset, predictions, tmp_path / 'report.json')
    assert len(result['iid']['groups']) == 2
    assert len(result['branches']['concat']['critical_strata']) == 14
    assert result['iid']['comparisons']['oracle_to_simulator']['language_mean_tv'] == 0
    assert result['iid']['comparisons']['concat_to_oracle']['joint_mean_tv'] == 0
    assert result['branches']['concat']['gates']['numeric_combined'] == 'not_run_registered_dataset'
    assert result['numeric_oracle_gate'] == 'not_run_registered_dataset'
    assert result['rare_contexts']['status'] == 'not_established'
    assert result == json.loads((tmp_path / 'report.json').read_text())


def test_prediction_join_rejects_wrong_ids_and_branch_set(smoke, tmp_path):
    dataset, _, predictions = smoke
    predictions['gru'].pop(next(iter(predictions['gru'])))
    with pytest.raises(ValueError, match='record ID'):
        evaluator().evaluate_seed(dataset, predictions, tmp_path / 'report.json')
    predictions.pop('gru')
    with pytest.raises(ValueError, match='branches'):
        evaluator().evaluate_seed(dataset, predictions, tmp_path / 'report.json')


def test_paraphrase_invalid_penalty_is_per_group_and_per_stratum(smoke, tmp_path):
    dataset, rows, predictions = smoke
    pair = next(row for row in rows if row['kind'] == 'paraphrase' and row['stratum'] == 'new_templates')
    predictions['concat'][pair['sides'][0]['record_id']] = (EOS,)
    result = evaluator().evaluate_seed(dataset, predictions, tmp_path / 'report.json')
    strata = result['branches']['concat']['paraphrase_strata']
    assert strata['new_templates']['mean_tv'] == 1
    assert strata['new_templates']['status_disagreements'] == 1
    assert strata['new_names_compositions']['mean_tv'] == 0
    assert result['branches']['concat']['critical_strata']['paraphrase/new_templates']['errors'] == 1


def test_oracle_core_errors_are_scored_against_real_simulator():
    wrong = WorldSpec(Regime.DETERMINISTIC, WorldSpec.for_regime(Regime.DETERMINISTIC).noise, True)
    result = evaluator().score_history(encode_events(simple()), simple(), wrong)
    assert result['branch_to_oracle']['language'] == 0
    assert result['oracle_to_simulator']['language'] == .125
    assert result['branch_to_simulator']['language'] == .125
    assert result['oracle_to_simulator']['joint'] == .5


def test_both_invalid_paraphrases_have_zero_status_disagreement_but_tv_one(smoke, tmp_path):
    dataset, rows, predictions = smoke
    pair = next(row for row in rows if row['kind'] == 'paraphrase' and row['stratum'] == 'new_templates')
    for side in pair['sides']:
        predictions['concat'][side['record_id']] = (EOS,)
    result = evaluator().evaluate_seed(dataset, predictions, tmp_path / 'both-invalid.json')
    value = result['branches']['concat']['paraphrase_strata']['new_templates']
    assert value['status_disagreements'] == 0
    assert value['mean_tv'] == 1


def test_dataset_duplicate_sides_rejected_even_after_checksum_repair(smoke, tmp_path):
    from hashlib import sha256
    dataset, _, predictions = smoke
    path = dataset / 'evaluator/challenge.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['sides'][1]['record_id'] = rows[0]['sides'][0]['record_id']
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['evaluator/challenge.jsonl'] = sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='invalid dataset'):
        evaluator().evaluate_seed(dataset, predictions, tmp_path / 'report.json')
