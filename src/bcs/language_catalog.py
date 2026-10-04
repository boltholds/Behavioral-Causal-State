"""Candidate Russian catalog; human review is required before L1 training lock."""
from .generator import TEMPLATES, NAMES, CommandEvent, ValueEvent
from .journal import Message

TRAIN_NAMES = NAMES
HOLDOUT_NAMES = ('Ирис','Лотос','Кедр','Ясень','Топаз','Опал','Базальт','Гранит')
FAMILIES = TEMPLATES + (
    {
        'Observation':'Шаг {time}. Прибор {name}: наблюдаем {var}={value}.',
        'Unknown':'Шаг {time}. Для прибора {name} в этой записи нет нового измерения {var}.',
        'Executed':'Шаг {time}. Прибору {name} подали команду {action}.',
        'Forced':'Шаг {time}. У прибора {name} принудительно зафиксировали {var}={value}.',
        'Hypothetical':'Шаг {time}. Для прибора {name} только рассмотрели возможность команды {action}.',
        'Denied':'Шаг {time}. Новую предложенную команду {action} прибору {name} оставили без выполнения.',
    },
    {
        'Observation':'В момент {time} измерение у {name} показало {var}={value}.',
        'Unknown':'В момент {time} эта запись о {name} не содержит нового измерения {var}.',
        'Executed':'В момент {time} для {name} исполнили команду {action}.',
        'Forced':'В момент {time} у {name} установили {var}={value} внешним воздействием.',
        'Hypothetical':'В момент {time} для {name} обсуждали гипотетическое выполнение {action}.',
        'Denied':'В момент {time} от исполнения новой предложенной команды {action} для {name} отказались.',
    },
)


def render_catalog(events,names,family,explicit_time=True):
    if type(family) is not int or not 0 <= family < len(FAMILIES) or len(names) != 2 or len(set(names)) != 2:
        raise ValueError('invalid rendering configuration')
    messages=[]
    for event in events:
        template=FAMILIES[family][type(event).__name__]
        if not explicit_time:
            for prefix in ('На шаге {time} ','на шаге {time} ','Шаг {time}. ','В момент {time} '):
                template=template.replace(prefix,'')
            template=template.replace(', шаг {time}', '')
        fields={'time':event.time,'name':names[event.object_id]}
        if isinstance(event,CommandEvent):
            fields['action']=event.action.value
        else:
            fields['var']=event.variable.value
            if isinstance(event,ValueEvent):
                fields['value']=event.value
        text=template.format(**fields)
        messages.append(Message('user','speaker-0',text[0].upper()+text[1:]))
    return tuple(messages)


def catalog_wire():
    return {'schema':'language-catalog-v1','families':FAMILIES,'train_families':[0,1],'heldout_families':[2,3],
            'train_names':TRAIN_NAMES,'heldout_names':HOLDOUT_NAMES,'human_review':'pending',
            'implicit_time':'message order advances only on Executed/Forced; other events refer to current slice'}
