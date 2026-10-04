import json
from pathlib import Path
import pytest
from bcs.journal import Request, Journal, HashCollision, ConflictingAnswer, key_payload

FIXTURES = Path(__file__).resolve().parents[1] / 'experiments/fixtures/journal-key-v1.json'

@pytest.mark.parametrize('case', json.loads(FIXTURES.read_text())['cases'], ids=lambda c: c['id'])
def test_exact_golden_keys(case):
    request = Request.from_json(json.dumps(case['request'], ensure_ascii=False))
    key, payload = key_payload(request)
    assert key == case['expected_key']
    assert len(payload) == case['expected_payload_bytes']
    if 'expected_payload_hex' in case:
        assert payload.hex() == case['expected_payload_hex']

@pytest.mark.parametrize('bad', ['null', '1.0', 'NaN', '"\\ud800"', '{"a":1,"a":2}'])
def test_reject_invalid_json_before_hashing(bad):
    request = json.loads(FIXTURES.read_text())['cases'][0]['request']
    wire = json.dumps(request).replace('"horizon": 1', '"horizon": ' + bad)
    with pytest.raises(ValueError):
        Request.from_json(wire)


def test_replay_is_persistent_and_append_only(tmp_path):
    from bcs.contracts import answer_from_wire
    request = Request.from_json(json.dumps(json.loads(FIXTURES.read_text())['cases'][0]['request']))
    from test_contracts import valid_answer
    answer = valid_answer()
    path = tmp_path / 'journal.sqlite'
    with Journal(path) as journal:
        assert journal.get(request) is None  # storage miss, not a domain state
        journal.put(request, answer)
        journal.put(request, answer)
        changed = answer_from_wire({**answer.to_wire(), 'snapshot_id': 'different'})
        with pytest.raises(ValueError):
            journal.put(request, changed)
    with Journal(path) as journal:
        assert journal.get(request) == answer


def test_collision_never_replays_another_payload(tmp_path):
    from test_contracts import valid_answer
    request = Request.from_json(json.dumps(json.loads(FIXTURES.read_text())['cases'][0]['request']))
    with Journal(tmp_path / 'j.db') as journal:
        journal.put(request, valid_answer())
        journal.connection.execute('UPDATE answers SET payload=?', (b'corrupted',))
        with pytest.raises(HashCollision):
            journal.get(request)

@pytest.mark.parametrize('field,value', [('horizon',True),('horizon',-1),('horizon',2),('future_actions',['fly']),('outcome',['Y','Y']),('object_id','device-99'),('unexpected',1)])
def test_journal_rejects_invalid_supported_query_ast(field,value):
    request = json.loads(FIXTURES.read_text())['cases'][0]['request']
    request['query'][field] = value
    with pytest.raises(ValueError):
        Request.from_json(json.dumps(request))


def test_conflicting_answer_requires_new_snapshot(tmp_path):
    from test_contracts import valid_answer
    from bcs.contracts import Point
    request = Request.from_json(json.dumps(json.loads(FIXTURES.read_text())['cases'][0]['request']))
    with Journal(tmp_path/'j.db') as journal:
        journal.put(request,valid_answer())
        with pytest.raises(ConflictingAnswer):
            journal.put(request,valid_answer(result=Point(.8)))
        assert journal.get(request) == valid_answer()
