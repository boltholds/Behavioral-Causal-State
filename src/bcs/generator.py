"""SPEC-04 base histories. Full challenge dataset/locked split manifest is separate.

Never hand GeneratedHistory to a learner: public_record is the export boundary.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import random
from .journal import Message
from .simulator import Action, Variable, Clamp, Step, WorldSpec, PhysicalTrace, simulate, bit, object_id
from .inference import Evidence, History

@dataclass(frozen=True)
class Event:
    time: int
    object_id: int
    def __post_init__(self):
        if type(self.time) is not int or self.time < 0:
            raise ValueError('invalid event time')
        object_id(self.object_id)

@dataclass(frozen=True)
class ValueEvent(Event):
    variable: Variable
    value: int
    def __post_init__(self):
        super().__post_init__()
        Clamp(self.object_id,self.variable,self.value)

@dataclass(frozen=True)
class Observation(ValueEvent):
    pass

@dataclass(frozen=True)
class Unknown(Event):
    variable: Variable
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.variable,Variable):
            raise ValueError('Variable enum required')

@dataclass(frozen=True)
class CommandEvent(Event):
    action: Action
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.action,Action) or self.action not in (Action.START,Action.STOP):
            raise ValueError('start/stop required')

@dataclass(frozen=True)
class Executed(CommandEvent):
    pass

@dataclass(frozen=True)
class Hypothetical(CommandEvent):
    pass

@dataclass(frozen=True)
class Denied(CommandEvent):
    pass

@dataclass(frozen=True)
class Forced(ValueEvent):
    pass

GroundedEvent = Observation | Unknown | Executed | Hypothetical | Denied | Forced


def event_wire(event):
    base = {'time':event.time,'entity':f'device-{event.object_id}'}
    if isinstance(event,CommandEvent):
        return base | {'mode':type(event).__name__, 'predicate':'command', 'negation':'execution' if isinstance(event,Denied) else 'none', 'value':event.action.value}
    if isinstance(event,(ValueEvent,Unknown)):
        return base | {'mode':type(event).__name__, 'predicate':event.variable.value, 'negation':'none', 'value':{'kind':'Unknown'} if isinstance(event,Unknown) else {'kind':'Binary','value':event.value}}
    raise ValueError('unknown event variant')


def compile_history(events):
    transitions = {}
    observations = []
    for e in events:
        if isinstance(e,Executed):
            step = Step.command(e.object_id,e.action)
        elif isinstance(e,Forced):
            step = Step((Action.HOLD,Action.HOLD),(Clamp(e.object_id,e.variable,e.value),))
        elif type(e) is Observation:
            observations.append(Evidence(e.time,e.object_id,e.variable,e.value))
            continue
        elif isinstance(e,(Hypothetical,Denied,Unknown)):
            continue
        else:
            raise ValueError('unknown event variant')
        if e.time < 1 or e.time in transitions:
            raise ValueError('one executed event per positive clock tick required')
        transitions[e.time] = step
    if set(transitions) != set(range(1,len(transitions)+1)):
        raise ValueError('missing physical transition')
    if any(e.time > len(transitions) for e in events):
        raise ValueError('event outside physical horizon')
    return History(tuple(transitions[t] for t in sorted(transitions)),tuple(sorted(observations,key=lambda e:(e.time,e.object_id,e.variable.value,e.value))))


# Two controlled template families; their independence/representativeness is NOT
# a validated natural-language benchmark. Every sentence names its physical time.
TEMPLATES = (
    {
        'Observation':'На шаге {time} у устройства {name} значение {var} равно {value}.',
        'Unknown':'В этой записи на шаге {time} новое измерение {var} устройства {name} не приведено.',
        'Executed':'На шаге {time} на устройстве {name} выполнили команду {action}.',
        'Forced':'На шаге {time} принудительно установили {var} устройства {name} в {value}.',
        'Hypothetical':'На шаге {time} рассмотрели вариант выполнения {action} на устройстве {name}.',
        'Denied':'В этой записи на шаге {time} зафиксирован отказ от выполнения ещё одной команды {action} на устройстве {name}.',
    },
    {
        'Observation':'Устройство {name}, шаг {time}: измеренное значение {var} — {value}.',
        'Unknown':'Устройство {name}, шаг {time}: эта запись не добавляет измерения {var}.',
        'Executed':'Устройство {name}, шаг {time}: выполнена команда {action}.',
        'Forced':'Устройство {name}, шаг {time}: значение {var} принудительно задано равным {value}.',
        'Hypothetical':'Устройство {name}, шаг {time}: обсуждалась возможность команды {action}.',
        'Denied':'Устройство {name}, шаг {time}: ещё одна предложенная команда {action} оставлена без выполнения.',
    },
)
NAMES = ('Альфа','Бета','Гамма','Дельта','Омега','Сигма','Каппа','Лямбда')


def render(events,names,family=0):
    if family not in (0,1) or len(names) != 2 or names[0] == names[1]:
        raise ValueError('valid template family and distinct names required')
    result = []
    for e in events:
        fields = dict(time=e.time,name=names[e.object_id])
        if isinstance(e,CommandEvent):
            fields['action'] = e.action.value
        else:
            fields['var'] = e.variable.value
            if isinstance(e,ValueEvent):
                fields['value'] = e.value
        text = TEMPLATES[family][type(e).__name__].format(**fields)
        result.append(Message('user','speaker-0',text))
    return tuple(result)

@dataclass(frozen=True)
class GeneratedHistory:
    group_id: str
    events: tuple[GroundedEvent,...]
    names: tuple[str,str]
    messages: tuple[Message,...]
    trace: PhysicalTrace


def _rng(seed,episode,channel):
    return random.Random(int.from_bytes(sha256(json.dumps([seed,episode,channel],ensure_ascii=False).encode()).digest(),'big'))


def generate(spec: WorldSpec,seed: int,episode_id: str,length=0):
    rng = _rng(seed,episode_id,'event-plan')
    surface = _rng(seed,episode_id,'surface')
    if length == 0:
        length = rng.choice((4,8,12,16))
    if length not in (4,8,12,16):
        raise ValueError('length must be 4,8,12,16')
    trace = simulate(spec,episode_id,seed,())
    events = [Observation(0,obj,Variable.C,trace.states[0][obj].c) for obj in range(2)]
    forced_position = rng.randrange(2,length)
    steps = []
    for position in range(2,length):
        obj = rng.randrange(2)
        kind = 'command' if position == forced_position else rng.choices(('command','clamp','observe','hypothetical','denied'),(.3,.2,.3,.1,.1))[0]
        time = len(steps)
        if kind == 'command':
            action = Action.START if rng.random() < (.8 if trace.states[-1][obj].c else .2) else Action.STOP
            steps.append(Step.command(obj,action))
            events.append(Executed(time+1,obj,action))
        elif kind == 'clamp':
            var, value = rng.choice((Variable.R,Variable.M,Variable.C)),rng.randrange(2)
            steps.append(Step((Action.HOLD,Action.HOLD),(Clamp(obj,var,value),)))
            events.append(Forced(time+1,obj,var,value))
        elif kind == 'observe':
            var = rng.choice(spec.visible)
            events.append(Unknown(time,obj,var) if rng.random() < .1 else Observation(time,obj,var,trace.states[-1][obj].get(var)))
        else:
            event = Hypothetical if kind == 'hypothetical' else Denied
            events.append(event(time,obj,rng.choice((Action.START,Action.STOP))))
        if len(steps) != time:
            trace = simulate(spec,episode_id,seed,tuple(steps))
    names = tuple(surface.sample(NAMES,2))
    events = tuple(events)
    return GeneratedHistory(episode_id,events,names,render(events,names,surface.randrange(2)),trace)


def semantic_key(events):
    """Evaluator-only duplicate detection. Never used by AnswerJournal."""
    return sha256(json.dumps([event_wire(e) for e in events],sort_keys=True,separators=(',',':')).encode()).hexdigest()


def public_record(history,include_labels=False):
    record = {'group_id':history.group_id,'messages':[{'role':m.role,'speaker_id':m.speaker_id,'text':m.text} for m in history.messages]}
    if include_labels:
        record['labels'] = [event_wire(e) for e in history.events]
    return record
