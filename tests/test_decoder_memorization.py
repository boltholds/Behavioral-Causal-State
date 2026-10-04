"""Catch false memorization, target leakage and invalid experimental controls."""
import json
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')
from experiments.diagnostics import decoder_memorization as dm
from bcs.generator import Executed
from bcs.grounder import Grounder, ReaderKind
from bcs.language_tokens import encode_events
from bcs.simulator import Action

torch.set_num_threads(2)


def test_selection_is_fixed_length_balanced_and_requires_distinct_targets():
    # Reordering by label or allowing repeated labels would defeat the input control.
    lengths = (8, 4, 16, 12, 4, 8, 12, 16, 4)
    targets = tuple((i,) for i in range(9))
    assert dm.select_indices(lengths, targets) == (1, 4, 0, 5, 3, 6, 2, 7)
    repeated = list(targets); repeated[4] = repeated[1]
    assert dm.select_indices(lengths, repeated)[:2] == (1, 8)
    with pytest.raises(ValueError, match='distinct'):
        dm.select_indices(lengths[:-1], repeated[:-1])


def test_control_moves_every_input_without_changing_length():
    lengths = (4, 4, 8, 8, 12, 12, 16, 16)
    targets = tuple((i,) for i in range(8))
    assert dm.same_length_permutation(lengths, targets) == (1, 0, 3, 2, 5, 4, 7, 6)
    with pytest.raises(ValueError):
        dm.same_length_permutation((4, 8), ((1,), (2,)))
    with pytest.raises(ValueError):
        dm.same_length_permutation((4, 4), ((1,), (1,)))


def test_missing_eos_counts_as_termination_and_history_failure():
    target = encode_events((Executed(1, 0, Action.START),))
    report = dm.sequence_diagnostics((target[:-1],), (target,))
    assert report['full_history_error'] == 1
    assert report['invalid_decodings'] == 1
    assert report['termination_errors'] == 1
    assert report['first_error_fields'] == {'termination': 1}
    exact = dm.sequence_diagnostics((target,), (target,))
    assert exact['full_history_error'] == 0
    assert exact['termination_errors'] == 0
    assert exact['first_error_fields'] == {}


@pytest.mark.parametrize('kind', list(ReaderKind))
def test_teacher_diagnostic_matches_incremental_same_prefix(kind):
    # Catches off-by-one teacher targets, wrong recurrent state or missing context.
    torch.manual_seed(11)
    model = Grounder(kind, 8).eval()
    z = torch.randn(2, 16, 8); lengths = torch.tensor([4, 8])
    targets = (
        encode_events((Executed(1, 0, Action.START),)),
        encode_events((Executed(1, 0, Action.STOP), Executed(2, 1, Action.START))),
    )
    parity = dm.teacher_step_parity(model, z, lengths, targets)
    assert parity['max_abs_valid_logit_difference'] < 1e-5
    assert parity['argmax_disagreements'] == 0
    measured = dm.measure(model, z, lengths, targets)
    assert measured['teacher']['tokens'] == 20
    assert measured['teacher']['nll'] > 0
    assert measured['free']['histories'] == 2


def test_success_requires_consecutive_checks_and_resets_on_failure():
    gate = dm.SuccessStreak(3)
    assert not gate.observe(True)
    assert not gate.observe(True)
    assert not gate.observe(False)
    assert not gate.observe(True)
    assert not gate.observe(True)
    assert gate.observe(True)


@pytest.mark.parametrize('values', [{'max_steps':0}, {'evaluate_every':0}, {'required_successes':0}, {'learning_rate':float('nan')}, {'seed':-1}])
def test_invalid_diagnostic_config_rejected(values):
    with pytest.raises(ValueError):
        dm.FitConfig(**values)


def test_budget_exhaustion_and_restored_prediction_provenance(tmp_path):
    torch.manual_seed(3)
    z = torch.randn(1, 16, 8); lengths = torch.tensor([4])
    targets = (encode_events((Executed(1, 0, Action.START),)),)
    provenance = {'fixture': 'unit-test', 'scope': 'train_set_memorization_diagnostic'}
    config = dm.FitConfig(max_steps=2, evaluate_every=1, required_successes=3)
    report = dm.fit(ReaderKind.GRU, z, lengths, targets, tmp_path/'fit', config, provenance)
    assert report['status'] == 'budget_exhausted'
    assert report['optimizer_steps'] == 2
    assert report['L1_status'] == 'not_run'
    assert report['generalization_evaluation'] == 'not_run'
    assert [x['step'] for x in report['history']] == [0, 1, 2]
    restored = dm.restore_fit(tmp_path/'fit', provenance)
    predictions = json.loads((tmp_path/'fit/predictions.json').read_text())
    assert [list(t) for t in restored.greedy(z, lengths)] == predictions['original']
    assert report['controls']['same_length_swap']['status'] == 'not_applicable_single_history'
    with pytest.raises(ValueError):
        dm.restore_fit(tmp_path/'fit', {'fixture': 'wrong'})
    with (tmp_path/'fit/model.pt').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError):
        dm.restore_fit(tmp_path/'fit', provenance)


def test_cli_refuses_unlocked_input_before_creating_output(tmp_path):
    result = subprocess.run([
        sys.executable, '-m', 'experiments.diagnostics.decoder_memorization',
        '--dataset', str(tmp_path), '--cache-root', str(tmp_path/'cache'),
        '--baseline-runs', str(tmp_path/'old'), '--output', str(tmp_path/'new'),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'input-lock' in result.stderr
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('changed', ['diagnostic_source', 'protocol', 'runtime_source'])
def test_restore_rejects_changed_current_implementation(tmp_path, monkeypatch, changed):
    # Historical expected provenance must not authorize running weights under changed code.
    import shutil
    from pathlib import Path
    import bcs.language_lock as runtime_lock
    z = torch.zeros(1, 16, 8); lengths = torch.tensor([4])
    targets = (encode_events((Executed(1, 0, Action.START),)),)
    provenance = {'fixture': 'unchanged-historical-provenance'}
    dm.fit(ReaderKind.GRU, z, lengths, targets, tmp_path/'fit',
           dm.FitConfig(max_steps=1, evaluate_every=1), provenance)
    if changed == 'diagnostic_source':
        altered = tmp_path/'changed.py'
        altered.write_text(Path(dm.__file__).read_text() + '\n# implementation changed\n')
        monkeypatch.setattr(dm, '__file__', str(altered))
    elif changed == 'protocol':
        altered = tmp_path/'protocol.json'
        altered.write_text(dm.PROTOCOL.read_text() + '\n')
        monkeypatch.setattr(dm, 'PROTOCOL', altered)
    else:
        copied = tmp_path/'runtime'; copied.mkdir()
        for source in Path(runtime_lock.__file__).parent.glob('*.py'):
            shutil.copyfile(source, copied/source.name)
        with (copied/'grounder.py').open('a') as f:
            f.write('\n# runtime implementation changed\n')
        monkeypatch.setattr(runtime_lock, '__file__', str(copied/'language_lock.py'))
    with pytest.raises(ValueError, match='implementation'):
        dm.restore_fit(tmp_path/'fit', provenance)


@pytest.mark.parametrize('key,value', [
    ('history_lengths', [4]), ('histories_per_length', 3), ('device', 'cuda'),
    ('selection', 'choose_easiest'), ('controls', []), ('L1_status', 'passed'),
])
def test_protocol_cannot_claim_settings_the_runner_ignores(key, value):
    protocol = json.loads(dm.PROTOCOL.read_text())
    protocol[key] = value
    with pytest.raises(ValueError, match='protocol'):
        dm.validate_protocol(protocol)
