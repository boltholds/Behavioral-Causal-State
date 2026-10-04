import json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from bcs.embedding_cache import build_cache,load_cache
from bcs.generator import Executed,event_wire
from bcs.simulator import Action
from bcs.language_training import load_labels,TrainConfig,train_one,load_checkpoint
from bcs.language_evaluation import structural_report,causal_report
from bcs.grounder import Grounder,ReaderKind
from bcs.language_tokens import encode_events,EOS
from bcs.calibration import calibration_panel,learn_core
from bcs.simulator import WorldSpec,Regime

torch.set_num_threads(2)

class FixtureEncoder:
    dimension=8;identity={'family':'unit-test-fixture'}
    def encode(self,texts):
        return np.array([[float('start' in t),float('stop' in t),1,0,0,0,0,0] for t in texts],dtype=np.float32)


def data(tmp_path,name):
    root=tmp_path/name;root.mkdir()
    rows=[];labels=[]
    for i in range(4):
        action=Action.START if i%2 else Action.STOP;rid=f'{name}-{i}:0'
        rows.append({'record_id':rid,'group_id':f'{name}-{i}','messages':[{'role':'user','speaker_id':'speaker-0','text':action.value}]})
        labels.append({'record_id':rid,'events':[event_wire(Executed(1,0,action))]})
    source=root/'public.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    lab=root/'labels.jsonl';lab.write_text(''.join(json.dumps(r)+'\n' for r in labels))
    build_cache(source,FixtureEncoder(),root/'cache')
    cache=load_cache(root/'cache',source)
    return cache,load_labels(lab,cache),lab


def test_label_join_requires_exact_train_id_set(tmp_path):
    a,_,_=data(tmp_path,'train');_,_,other=data(tmp_path,'validation')
    with pytest.raises(ValueError):load_labels(other,a)


def test_train_restore_is_deterministic_and_never_claims_l1(tmp_path):
    a,ta,_=data(tmp_path,'train');b,tb,_=data(tmp_path,'validation')
    config=TrainConfig(epochs=2,batch_size=4,learning_rate=.001,seed=11,patience=5)
    reports=[]
    for name in ('run1','run2'):
        report=train_one(ReaderKind.GRU,a,ta,b,tb,tmp_path/name,config,input_lock_sha256="0"*64)
        assert report['L1_status']=='not_run_diagnostic'
        model=load_checkpoint(tmp_path/name,a.manifest['encoder'],"0"*64)
        z,n=b.batch(range(4));pred=model.greedy(torch.from_numpy(z),torch.from_numpy(n))
        reports.append((report['history'],pred))
    assert reports[0]==reports[1]
    with (tmp_path/'run1/model.pt').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'run1',a.manifest['encoder'],"0"*64)


def test_structural_errors_count_missing_events_and_false_unknown():
    gold=(encode_events((Executed(1,0,Action.START),)),)
    report=structural_report(((EOS,),),gold)
    assert report['full_history_error']==1 and report['full_event_error']==1
    assert all(x['errors']==1 for x in report['fields'].values())


def test_invalid_predictions_have_causal_penalty_one():
    spec=WorldSpec.for_regime(Regime.DETERMINISTIC);core=learn_core(calibration_panel(spec).probes)
    truth=((Executed(1,0,Action.START),),)
    report=causal_report(((EOS,),),truth,core)
    assert report['language_mean_tv']==1 and report['joint_mean_tv']==1
    assert report['invalid_histories']==1


def test_restoration_rejects_edited_checkpoint_provenance(tmp_path):
    a,ta,_=data(tmp_path,'train');b,tb,_=data(tmp_path,'validation')
    train_one(ReaderKind.GRU,a,ta,b,tb,tmp_path/'run',TrainConfig(epochs=1,batch_size=4),input_lock_sha256="0"*64)
    path=tmp_path/'run/checkpoint.json';metadata=json.loads(path.read_text())
    metadata['protocol_scope']='registered';path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):load_checkpoint(tmp_path/'run',a.manifest['encoder'],"0"*64)
