import json
from pathlib import Path

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from experiments.l1 import training as training
from bcs.embedding_cache import build_cache,load_cache
from bcs.generator import Executed,event_wire
from bcs.grounder import ReaderKind
from bcs.language_training import TrainConfig,load_labels
from bcs.simulator import Action

torch.set_num_threads(1)


class Encoder:
    dimension=4
    identity={'family':'test-only'}
    def encode(self,texts):
        return np.array([[float(t=='start'),float(t=='stop'),1.,0.] for t in texts],dtype=np.float32)


def split(tmp_path,name):
    root=tmp_path/name;root.mkdir()
    records=[];labels=[]
    for i in range(4):
        action=Action.START if i%2 else Action.STOP
        records.append({'record_id':f'{name}-{i}','group_id':f'{name}-{i}','messages':[{'role':'user','speaker_id':'speaker-0','text':action.value}]})
        labels.append({'record_id':f'{name}-{i}','events':[event_wire(Executed(1,0,action))]})
    (root/'public.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (root/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
    build_cache(root/'public.jsonl',Encoder(),root/'cache')
    cache=load_cache(root/'cache',root/'public.jsonl')
    return cache,load_labels(root/'labels.jsonl',cache)


def inputs(tmp_path):
    a,ta=split(tmp_path,'train');b,tb=split(tmp_path,'validation')
    return a,ta,b,tb


def test_resume_reproduces_uninterrupted_weights_and_metrics(tmp_path):
    args=inputs(tmp_path)
    config=TrainConfig(epochs=2,batch_size=2,learning_rate=.001,seed=11)
    identity={'study':'fixture','job':'gru/lr.001/seed11'}
    full=training.fit_run(ReaderKind.GRU,*args,tmp_path/'full',config,identity)
    paused=training.fit_run(ReaderKind.GRU,*args,tmp_path/'resumed',config,identity,max_new_epochs=1)
    assert paused['status']=='paused' and paused['completed_epochs']==1
    with pytest.raises(ValueError,match='completed'):
        training.load_completed_run(tmp_path/'resumed',identity)
    torch.rand(17)  # Ambient RNG use must not perturb resumed dropout or shuffling.
    resumed=training.fit_run(ReaderKind.GRU,*args,tmp_path/'resumed',config,identity)
    assert resumed['status']=='completed'
    assert resumed['committed_optimizer_steps']==4
    assert resumed['history']==full['history']
    assert resumed['best_epoch']==full['best_epoch']
    x=training.load_completed_run(tmp_path/'full',identity)
    y=training.load_completed_run(tmp_path/'resumed',identity)
    assert all(torch.equal(x.state_dict()[k],y.state_dict()[k]) for k in x.state_dict())
    # A rerun must not spend another epoch or mutate the completed checkpoint.
    before=(tmp_path/'resumed/best.pt').read_bytes()
    assert training.fit_run(ReaderKind.GRU,*args,tmp_path/'resumed',config,identity)['history']==resumed['history']
    assert (tmp_path/'resumed/best.pt').read_bytes()==before


def test_resume_refuses_changed_config_inputs_or_identity(tmp_path):
    args=inputs(tmp_path);config=TrainConfig(epochs=2,batch_size=2)
    identity={'job':'original'}
    training.fit_run(ReaderKind.CONCAT,*args,tmp_path/'run',config,identity,max_new_epochs=1)
    with pytest.raises(ValueError):
        training.fit_run(ReaderKind.CONCAT,*args,tmp_path/'run',TrainConfig(epochs=3,batch_size=2),identity)
    with pytest.raises(ValueError):
        training.fit_run(ReaderKind.CONCAT,*args,tmp_path/'run',config,{'job':'other'})
    a,ta,b,tb=args
    with pytest.raises(ValueError):
        training.fit_run(ReaderKind.CONCAT,a,tuple(reversed(ta)),b,tb,tmp_path/'run',config,identity)


def test_completed_checkpoint_corruption_is_rejected(tmp_path):
    args=inputs(tmp_path);identity={'job':'fixture'}
    training.fit_run(ReaderKind.GRU,*args,tmp_path/'run',TrainConfig(epochs=1,batch_size=4),identity)
    with (tmp_path/'run/best.pt').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError):training.load_completed_run(tmp_path/'run',identity)


def test_training_never_accepts_overlapping_validation_groups(tmp_path):
    a,t=split(tmp_path,'same')
    with pytest.raises(ValueError,match='overlap'):
        training.fit_run(ReaderKind.GRU,a,t,a,t,tmp_path/'run',TrainConfig(epochs=1),{'job':'bad'})


def test_train_job_alone_never_claims_passed_l1(tmp_path):
    args=inputs(tmp_path)
    result=training.fit_run(ReaderKind.ATTENTION,*args,tmp_path/'run',TrainConfig(epochs=1,batch_size=4),{'job':'fixture'})
    assert result['L1_status']=='not_evaluated'
    assert result['dtype']=='float32'
    assert result['resources']['process_peak_rss_bytes']>0
    assert result['resources']['inference_flops_estimate']['assumptions']


def test_patience_tracks_event_error_not_nll_only_improvements(tmp_path,monkeypatch):
    args=inputs(tmp_path);calls=[]
    def validation(*args):
        calls.append(1)
        return {'full_event_error':1.,'full_history_error':1.,'teacher_forced_nll':1./len(calls)}
    monkeypatch.setattr(training,'validation_metrics',validation)
    result=training.fit_run(ReaderKind.GRU,*args,tmp_path/'run',TrainConfig(epochs=10,batch_size=4,patience=2),{'job':'patience'})
    assert result['completed_epochs']==3
    assert result['stop_reason']=='patience'
    assert result['best_epoch']==3  # NLL remains the checkpoint tie-break, not the patience trigger.


def test_editing_completed_validation_score_cannot_change_selection(tmp_path):
    args=inputs(tmp_path);identity={'job':'bound-report'}
    training.fit_run(ReaderKind.GRU,*args,tmp_path/'run',TrainConfig(epochs=1,batch_size=4),identity)
    path=tmp_path/'run/report.json';report=json.loads(path.read_text());report['best_validation_event_error']=-1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):training.load_completed_run(tmp_path/'run',identity)
