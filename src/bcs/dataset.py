"""Deterministic artifact builder and audit for the L1 data preparation stage."""
from collections import Counter,defaultdict
from dataclasses import dataclass,asdict
from hashlib import sha256
from pathlib import Path
import json
from .generator import generate,compile_history,event_wire,Observation,Executed,_rng
from .language_catalog import render_catalog,catalog_wire,TRAIN_NAMES,HOLDOUT_NAMES
from .simulator import WorldSpec,Regime,simulate
from .calibration import calibration_panel,learn_core,BooleanCore
from .pairs import PairKind,HoldoutStratum,ChallengePair,PairSide,make_pair,verify_pair,implicit_clock_valid
from .dataset_codec import event_from_wire,annotation_wire,annotation_from_wire,panel_wire,panel_from_wire
from .journal import Message

SPLITS=('train','validation','iid_test','challenge')

@dataclass(frozen=True)
class DatasetCounts:
    train: int=20000
    validation: int=4000
    iid_test: int=4000
    pairs_per_stratum: int=1000
    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in asdict(self).values()):
            raise ValueError('positive integer dataset counts required')


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def semantic_fingerprint(events):
    # Explicit time fixes physical order; narrative reordering at the same time
    # is immaterial. Multiplicity is retained, surfaces/names are excluded.
    normalized=sorted((event_wire(e) for e in events),key=lambda e:(e['time'],_json(e)))
    return sha256(_json(normalized).encode()).hexdigest()


def _id(seed,split,index):
    return sha256(_json([seed,split,index]).encode()).hexdigest()[:32]


def _owner(key,seed):
    # Fixed ownership, independent of desired quotas/iteration order.
    bucket=int(sha256(f'{seed}:{key}'.encode()).hexdigest(),16)%7
    return 'train' if bucket<5 else 'validation' if bucket==5 else 'iid_test'


def _public(group,side,messages):
    return {'record_id':f'{group}:{side}','group_id':group,'messages':[asdict(m) for m in messages]}


def _dump(path,value):
    path.write_text(_json(value)+'\n')


def build_dataset(output,counts=DatasetCounts(),seed=20261004,progress=lambda value:None):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('dataset destination must be empty; refusing overwrite')
    if type(seed) is not int or seed<0:
        raise ValueError('nonnegative split seed required')
    for name in ('public','evaluator','training'):
        (output/name).mkdir(parents=True,exist_ok=True)
    spec=WorldSpec.for_regime(Regime.DETERMINISTIC)
    panel=calibration_panel(spec)
    core=learn_core(panel.probes)
    if not isinstance(core,BooleanCore):
        raise ValueError('calibration did not establish a complete core')
    _dump(output/'calibration.json',panel_wire(panel)); _dump(output/'core.json',core.to_wire())
    _dump(output/'catalog.json',catalog_wire())
    ownership={}; rejected=Counter(); accepted=Counter(); length_counts=defaultdict(Counter)
    for split in SPLITS[:3]:
        target=getattr(counts,split); attempt=0
        with (output/f'public/{split}.jsonl').open('w') as pub,(output/f'evaluator/{split}.jsonl').open('w') as private:
            while accepted[split]<target:
                if attempt>=max(10000,target*200):
                    raise RuntimeError(f'support exhausted for {split}')
                group=_id(seed,split,attempt); attempt+=1
                length=_rng(seed,f'{split}:{accepted[split]}','requested-length').choice((4,8,12,16))
                history=generate(spec,seed,group,length)
                key=semantic_fingerprint(history.events)
                if _owner(key,seed)!=split:
                    rejected[split]+=1; continue
                ownership[key]=split
                surface=_rng(seed,group,'dataset-render')
                family=surface.randrange(2); explicit=bool(surface.randrange(2))
                messages=render_catalog(history.events,history.names,family,explicit)
                pub.write(_json(_public(group,0,messages))+'\n')
                row={'group_id':group,'seed':seed,'kind':'base','stratum':'iid','names':history.names,'families':[family],
                     'explicit_time':explicit,'sides':[{'record_id':f'{group}:0','events':[event_wire(e) for e in history.events]}]}
                private.write(_json(row)+'\n')
                accepted[split]+=1; length_counts[split][len(history.events)]+=1
                if accepted[split]%1000==0:
                    progress({'split':split,'accepted':accepted[split],'target':target})
    # Expose supervised labels only for train through a separate named artifact.
    with (output/'training/labels.jsonl').open('w') as file:
        for line in (output/'evaluator/train.jsonl').open():
            row=json.loads(line)
            file.write(_json({'record_id':row['sides'][0]['record_id'],'events':row['sides'][0]['events']})+'\n')
    with (output/'public/challenge.jsonl').open('w') as pub,(output/'evaluator/challenge.jsonl').open('w') as private:
        for kind in PairKind:
            for stratum in HoldoutStratum:
                label=f'{kind.value}/{stratum.value}'; n=0; attempt=0
                while n<counts.pairs_per_stratum:
                    if attempt>=max(10000,counts.pairs_per_stratum*200):
                        raise RuntimeError(f'support exhausted for {label}')
                    group=_id(seed,label,attempt); attempt+=1
                    pair=make_pair(seed,group,kind,stratum)
                    keys=[semantic_fingerprint(s.events) for s in (pair.left,pair.right)]
                    if any(key in ownership and ownership[key]!='challenge' for key in keys):
                        rejected[label]+=1; continue
                    for key in keys:
                        ownership[key]='challenge'
                    row={'group_id':group,'seed':seed,'kind':kind.value,'stratum':stratum.value,'names':pair.names,
                         'families':pair.families,'explicit_time':pair.explicit_time,'sides':[], 'annotation':annotation_wire(pair.annotation)}
                    for side_index,side in enumerate((pair.left,pair.right)):
                        pub.write(_json(_public(group,side_index,side.messages))+'\n')
                        row['sides'].append({'record_id':f'{group}:{side_index}','events':[event_wire(e) for e in side.events]})
                    private.write(_json(row)+'\n'); n+=1; accepted[label]+=1
                progress({'split':'challenge','stratum':label,'accepted':n})
    files={str(p.relative_to(output)):sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*')) if p.is_file()}
    manifest={'schema':'l1-dataset-v1','seed':seed,'counts':asdict(counts),'mode':'registered_size' if counts==DatasetCounts() else 'smoke',
              'files':files,'core_artifact_id':core.artifact_id,'catalog_human_review':'pending',
              'accepted':dict(accepted),'rejected':dict(rejected),'base_length_counts':dict(length_counts),
              'split_policy':'fixed semantic hash ownership 5:1:1 for base; challenge rejects both sides intersecting base',
              'challenge_length_support':[8,12,16],'base_explicit_time_probability':.5,
              'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}}
    _dump(output/'manifest.json',manifest)
    return manifest


def audit_dataset(output,progress=lambda value:None):
    """Re-read actual public/private artifacts; no trust in claimed status/counts."""
    output=Path(output); errors=[]; owners={}; groups={}; base=Counter(); challenge=Counter(); public_count=Counter()
    diagnostics=defaultdict(Counter); semantic_duplicates=0; seen_records=set(); training_expected={}
    def error(message):
        if len(errors)<100:
            errors.append(message)
    try:
        manifest=json.loads((output/'manifest.json').read_text()); counts=DatasetCounts(**manifest['counts'])
        if manifest['schema']!='l1-dataset-v1':
            raise ValueError('unknown dataset schema')
        expected_mode='registered_size' if counts==DatasetCounts() else 'smoke'
        if manifest['mode']!=expected_mode or manifest['challenge_length_support']!=[8,12,16]:
            error('declared dataset mode or length support mismatch')
        expected_files={'calibration.json','core.json','catalog.json','training/labels.jsonl'}|{f'{folder}/{s}.jsonl' for folder in ('public','evaluator') for s in SPLITS}
        if set(manifest['files'])!=expected_files:
            error('manifest file set mismatch')
        actual_exports={str(p.relative_to(output)) for folder in ('public','evaluator','training') for p in (output/folder).rglob('*') if p.is_file()}
        if actual_exports!={name for name in expected_files if '/' in name}:
            error('undeclared or missing learner/evaluator export')
        for name in expected_files:
            if sha256((output/name).read_bytes()).hexdigest()!=manifest['files'].get(name):
                error(f'file hash mismatch: {name}')
        if json.loads((output/'catalog.json').read_text())!=json.loads(_json(catalog_wire())):
            error('catalog mismatch')
        panel=panel_from_wire(json.loads((output/'calibration.json').read_text()))
        core=BooleanCore.from_wire(json.loads((output/'core.json').read_text()))
        spec=WorldSpec.for_regime(Regime.DETERMINISTIC)
        if panel!=calibration_panel(spec):
            error('calibration differs from registered single-target probe schedule')
        for index,probe in enumerate(panel.probes):
            if any(len(step.clamps)>1 for step in probe.history.steps):
                error('joint clamp in single-target calibration')
            trace=simulate(spec,f'calibration-{index}',0,probe.history.steps)
            if any(e.value!=trace.states[e.time][e.object_id].get(e.variable) for e in probe.history.evidence):
                error('calibration evidence disagrees with simulator')
        if len(panel.probes)!=12 or panel.action_count!=22 or learn_core(panel.probes)!=core or core.artifact_id!=manifest['core_artifact_id']:
            error('calibration/core mismatch')
        for split in SPLITS:
            public={}
            for line in (output/f'public/{split}.jsonl').open():
                row=json.loads(line)
                if set(row)!= {'record_id','group_id','messages'}:
                    error(f'public fields violate learner boundary: {split}'); continue
                rid=row['record_id']
                if rid in seen_records:
                    error(f'duplicate record ID: {rid}')
                seen_records.add(rid); public[rid]=row; public_count[split]+=1
                support=(8,12,16) if split=='challenge' else (4,8,12,16)
                if len(row['messages']) not in support:
                    error('history length outside registered support')
                for message in row['messages']:
                    if set(message)!= {'role','speaker_id','text'}:
                        error('public message metadata leak')
            private_ids=set()
            for index,line in enumerate((output/f'evaluator/{split}.jsonl').open()):
                row=json.loads(line); group=row['group_id']
                if group in groups:
                    error(f'duplicate group ID: {group}')
                groups[group]=split
                if row['seed']!=manifest['seed'] or type(row['explicit_time']) is not bool:
                    error('seed/render mode mismatch')
                sides=[]
                try:
                    for side_index,side in enumerate(row['sides']):
                        events=tuple(event_from_wire(e) for e in side['events'])
                        rid=side['record_id']; private_ids.add(rid)
                        if rid!=f'{group}:{side_index}':
                            error('record/group mismatch')
                        published=public.get(rid)
                        if published is None or published['group_id']!=group:
                            error('missing public counterpart'); continue
                        messages=tuple(Message(**m) for m in published['messages'])
                        history=compile_history(events)
                        if not row['explicit_time'] and not implicit_clock_valid(events):
                            error('implicit clock disagrees with message order')
                        if len(events)!=len(messages) or not any(isinstance(e,Executed) for e in events):
                            error('invalid event/message count or missing executed command')
                        trace=simulate(WorldSpec.for_regime(Regime.DETERMINISTIC),group,row['seed'],history.steps)
                        if any(type(e) is Observation and e.value!=trace.states[e.time][e.object_id].get(e.variable) for e in events):
                            error('factual observation disagrees with simulator')
                        if messages!=render_catalog(events,tuple(row['names']),row['families'][side_index],row['explicit_time']):
                            error('public text disagrees with labels/catalog')
                        key=semantic_fingerprint(events)
                        if key in owners and owners[key]!=split:
                            semantic_duplicates+=1; error('cross-split semantic duplicate')
                        owners[key]=split
                        for e in events:
                            diagnostics[f'{split}:mode'][type(e).__name__]+=1
                            diagnostics[f'{split}:name_mode'][row['names'][e.object_id]+'/'+type(e).__name__]+=1
                            if type(e) is Observation:
                                diagnostics[f'{split}:name_value'][row['names'][e.object_id]+'/'+e.variable.value+'/'+str(e.value)]+=1
                        diagnostics[f'{split}:length'][str(len(events))]+=1
                        diagnostics[f'{split}:template'][str(row['families'][side_index])]+=1
                        sides.append(PairSide(events,messages))
                    if split=='challenge':
                        kind=PairKind(row['kind']); stratum=HoldoutStratum(row['stratum'])
                        challenge[f'{kind.value}/{stratum.value}']+=1
                        if len(sides)!=2:
                            error('pair needs two sides'); continue
                        allowed_names=TRAIN_NAMES if stratum==HoldoutStratum.TEMPLATES else HOLDOUT_NAMES
                        allowed_families=(2,3) if stratum==HoldoutStratum.TEMPLATES else (0,1)
                        if any(n not in allowed_names for n in row['names']) or any(f not in allowed_families for f in row['families']):
                            error('holdout catalog contamination')
                        pair=ChallengePair(group,row['seed'],kind,stratum,*sides,tuple(row['names']),tuple(row['families']),row['explicit_time'],annotation_from_wire(row['annotation']))
                        if not verify_pair(pair):
                            error('pair relation/witness failed verification')
                        diagnostics[f'challenge:{kind.value}:relation'][row['annotation']['kind']]+=1
                    else:
                        base[split]+=1
                        if row['kind']!='base' or row['stratum']!='iid' or len(sides)!=1 or any(f not in (0,1) for f in row['families']) or any(n not in TRAIN_NAMES for n in row['names']):
                            error('base catalog or shape mismatch')
                        if split=='train' and sides:
                            training_expected[row['sides'][0]['record_id']]=[event_wire(e) for e in sides[0].events]
                except (ValueError,KeyError,TypeError,IndexError) as exc:
                    error(f'invalid evaluator record {group}: {exc}')
                if (index+1)%2000==0:
                    progress({'audit_split':split,'groups':index+1})
            if private_ids!=set(public):
                error(f'public/private ID set mismatch: {split}')
        labels={}
        for line in (output/'training/labels.jsonl').open():
            row=json.loads(line)
            if set(row)!= {'record_id','events'} or row['record_id'] in labels:
                error('invalid/duplicate training labels')
            labels[row['record_id']]=row['events']
        if labels!=training_expected:
            error('training labels differ from train-only oracle labels')
        for split in SPLITS[:3]:
            if base[split]!=getattr(counts,split) or public_count[split]!=base[split]:
                error(f'base count mismatch: {split}')
            if dict(diagnostics[f'{split}:length'])!=manifest['base_length_counts'][split]:
                error(f'base length histogram mismatch: {split}')
        for kind in PairKind:
            for stratum in HoldoutStratum:
                if challenge[f'{kind.value}/{stratum.value}']!=counts.pairs_per_stratum:
                    error('challenge stratum count mismatch')
        if public_count['challenge']!=2*sum(challenge.values()):
            error('challenge public count mismatch')
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        error(f'invalid dataset: {exc}')
    return {'integrity_status':'failed' if errors else 'passed','training_ready':False,'training_blockers':['human catalog review','encoder/tokenizer manifest lock'],
            'base_counts':dict(base),'challenge_groups':sum(challenge.values()),'challenge_strata':dict(challenge),
            'cross_split_semantic_duplicates':semantic_duplicates,'diagnostics':dict(diagnostics),'errors':errors}
