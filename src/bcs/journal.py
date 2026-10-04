"""Exact request journal. No semantic canonicalization or fuzzy lookup."""
from dataclasses import dataclass
from hashlib import sha256
import json
import sqlite3
from .contracts import AnswerRecord, answer_from_wire
from .queries import decode_query_ast


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _validate(value):
    if type(value) is str:
        try:
            value.encode('utf-8')
        except UnicodeEncodeError as error:
            raise ValueError('unpaired surrogate') from error
    elif type(value) in (bool, int):
        return
    elif type(value) is list:
        for item in value:
            _validate(item)
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('object keys must be strings')
            _validate(key)
            _validate(item)
    else:
        raise ValueError('only string/bool/int/array/object allowed')


def canonical_json(value):
    _validate(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')

@dataclass(frozen=True)
class Message:
    role: str
    speaker_id: str
    text: str
    def __post_init__(self):
        for value in (self.role, self.speaker_id, self.text):
            if type(value) is not str:
                raise ValueError('message fields must be strings')
            _validate(value)

@dataclass(frozen=True)
class Request:
    snapshot_id: str
    messages: tuple[Message, ...]
    query_bytes: bytes

    def __post_init__(self):
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id:
            raise ValueError('snapshot_id required')
        _validate(self.snapshot_id)
        if not isinstance(self.messages, tuple) or not all(isinstance(m, Message) for m in self.messages):
            raise ValueError('immutable messages required')
        q = json.loads(self.query_bytes, object_pairs_hook=_object)
        if canonical_json(q) != self.query_bytes:
            raise ValueError('query bytes must use canonical serialization')
        decode_query_ast(q)

    @classmethod
    def from_json(cls, text):
        value = json.loads(text, object_pairs_hook=_object)
        _validate(value)
        if not isinstance(value, dict) or set(value) != {'snapshot_id', 'messages', 'query'}:
            raise ValueError('unexpected request fields')
        if not isinstance(value['messages'], list):
            raise ValueError('messages must be an array')
        messages = []
        for message in value['messages']:
            if not isinstance(message, dict) or set(message) != {'role', 'speaker_id', 'text'}:
                raise ValueError('unexpected message fields')
            messages.append(Message(**message))
        return cls(value['snapshot_id'], tuple(messages), canonical_json(value['query']))


def _seq(items):
    items = tuple(items)
    return len(items).to_bytes(8, 'big') + b''.join(len(x).to_bytes(8, 'big') + x for x in items)


def key_payload(request):
    history = _seq(_seq(s.encode('utf-8') for s in (m.role, m.speaker_id, m.text)) for m in request.messages)
    payload = b'BCS.AnswerJournal/v1\0' + _seq((request.snapshot_id.encode('utf-8'), history, request.query_bytes))
    return 'sha256:' + sha256(payload).hexdigest(), payload

class HashCollision(RuntimeError):
    pass

class ConflictingAnswer(RuntimeError):
    pass

class Journal:
    """SQLite append-only journal. None means storage miss, never a causal status."""
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, payload BLOB NOT NULL, answer TEXT NOT NULL)')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.connection.close()

    def get(self, request):
        key, payload = key_payload(request)
        row = self.connection.execute('SELECT payload,answer FROM answers WHERE key=?', (key,)).fetchone()
        if row is None:
            return None
        if bytes(row[0]) != payload:
            raise HashCollision(key)
        return answer_from_wire(json.loads(row[1]))

    def put(self, request, answer: AnswerRecord):
        if request.snapshot_id != answer.snapshot_id:
            raise ValueError('answer and request snapshots differ')
        key, payload = key_payload(request)
        encoded = json.dumps(answer.to_wire(), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        with self.connection:
            self.connection.execute('INSERT OR IGNORE INTO answers VALUES (?,?,?)', (key, payload, encoded))
            previous = self.get(request)
            if previous != answer:
                raise ConflictingAnswer('immutable snapshot already has another answer; create a new snapshot')
