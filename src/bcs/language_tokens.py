"""Finite structured language; no gold-dependent masks or implicit EOS repair."""
from dataclasses import dataclass
from .generator import GroundedEvent, event_wire, compile_history
from .dataset_codec import event_from_wire

PAD, BOS, EOS = 0, 1, 2
MAX_EVENTS = 16
MAX_TOKENS = 6 * MAX_EVENTS + 1
FIELDS = ('time', 'mode', 'entity', 'predicate', 'negation', 'value')
VALUES = (
    tuple(range(17)),
    ('Observation', 'Unknown', 'Executed', 'Forced', 'Hypothetical', 'Denied'),
    ('device-0', 'device-1'),
    ('R', 'M', 'C', 'Y', 'command'),
    ('none', 'execution'),
    ('0', '1', 'Unknown', 'start', 'stop'),
)
OFFSETS = (3, 20, 26, 28, 33, 35)
VOCAB_SIZE = 40


@dataclass(frozen=True)
class DecodedEvents:
    events: tuple[GroundedEvent, ...]


@dataclass(frozen=True)
class InvalidDecoding:
    reason: str


def _value(token, field):
    index = token - OFFSETS[field]
    if type(token) is not int or not 0 <= index < len(VALUES[field]):
        raise ValueError('token outside field vocabulary')
    return VALUES[field][index]


def _choices(prefix):
    pos = len(prefix) % 6
    start = len(prefix) - pos
    if pos == 0:
        return ((EOS,) if prefix else ()) + (tuple(range(3, 20)) if start < 96 else ())
    choices = VALUES[pos]
    if pos >= 3:
        mode = _value(prefix[start + 1], 1)
        command = mode in ('Executed', 'Hypothetical', 'Denied')
        if pos == 3:
            choices = ('command',) if command else ('R', 'M', 'C', 'Y')
        elif pos == 4:
            choices = ('execution',) if mode == 'Denied' else ('none',)
        elif pos == 5:
            choices = ('start', 'stop') if command else ('Unknown',) if mode == 'Unknown' else ('0', '1')
    return tuple(OFFSETS[pos] + VALUES[pos].index(v) for v in choices)


def allowed_tokens(prefix):
    prefix = tuple(prefix)
    for i, token in enumerate(prefix):
        if type(token) is not int or token not in _choices(prefix[:i]):
            raise ValueError('invalid structured prefix')
        if token == EOS:
            if i != len(prefix) - 1:
                raise ValueError('token after EOS')
            return ()
    return _choices(prefix)


def encode_events(events):
    events = tuple(events)
    if not 1 <= len(events) <= MAX_EVENTS:
        raise ValueError('one to sixteen events required')
    compile_history(events)
    result = []
    for event in events:
        wire = event_wire(event)
        value = wire['value']
        wire['value'] = str(value['value']) if isinstance(value, dict) and value['kind'] == 'Binary' else 'Unknown' if isinstance(value, dict) else value
        for j, field in enumerate(FIELDS):
            result.append(OFFSETS[j] + VALUES[j].index(wire[field]))
    return tuple(result) + (EOS,)


def decode_events(tokens):
    tokens = tuple(tokens)
    try:
        if not tokens or tokens[-1] != EOS or len(tokens) > MAX_TOKENS:
            raise ValueError('missing EOS or sequence too long')
        allowed_tokens(tokens)
        events = []
        for i in range(0, len(tokens)-1, 6):
            wire = {field: _value(tokens[i+j], j) for j, field in enumerate(FIELDS)}
            value = wire['value']
            if value in ('0', '1'):
                wire['value'] = {'kind': 'Binary', 'value': int(value)}
            elif value == 'Unknown':
                wire['value'] = {'kind': 'Unknown'}
            events.append(event_from_wire(wire))
        compile_history(events)
        return DecodedEvents(tuple(events))
    except (ValueError, IndexError, TypeError) as error:
        return InvalidDecoding(str(error))
