"""Study sequencing, held-out isolation and review gates, independent of model quality."""
import json
from pathlib import Path
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from bcs.dataset import build_dataset,DatasetCounts
from experiments.l1.preparation import prepare
from experiments.l1 import study


class FakeEncoder:
    dimension=4
    identity={'family':'fixture'}
    def encode(self,texts):return np.ones((len(texts),4),dtype=np.float32)


@pytest.fixture
def ready(tmp_path):
    dataset=tmp_path/'data';build_dataset(dataset,DatasetCounts(2,2,2,1),seed=114)
    cache=tmp_path/'cache';prepare(dataset,FakeEncoder(),cache)
    root=tmp_path/'study'
    study.initialize(dataset,cache,root,code_revision='0'*40,scope='fixture',fixture_epochs=1)
    return root


def test_fixture_cannot_claim_registered_and_early_test_is_rejected(ready):
    state=study.status(ready)
    assert state['L1_status']=='not_run_fixture'
    with pytest.raises(ValueError,match='final'):
        study.evaluate_study(ready)
    with pytest.raises(ValueError,match='tuning'):
        study.freeze_selection(ready)


def test_selection_needs_all_nine_completed_trials_and_uses_predeclared_tie_break(ready,monkeypatch):
    from experiments.l1 import training
    monkeypatch.setattr(training,'validation_metrics',lambda *args:{'full_event_error':0.,'full_history_error':0.,'teacher_forced_nll':1.})
    assert study.train_stage(ready,'tuning',max_jobs=8)['completed_jobs']==8
    with pytest.raises(ValueError,match='tuning'):study.freeze_selection(ready)
    study.train_stage(ready,'tuning')
    selection=study.freeze_selection(ready)
    assert all(v['learning_rate']==.0001 for v in selection['branches'].values())
    before=(ready/'selection.json').read_bytes()
    assert study.freeze_selection(ready)==selection
    assert (ready/'selection.json').read_bytes()==before
    altered=json.loads(before);altered['branches']['gru']['learning_rate']=.01
    study.atomic_json(ready/'selection.json',altered)
    with pytest.raises(ValueError,match='selection changed'):study.train_stage(ready,'final')
    (ready/'selection.json').write_bytes(before)
    with pytest.raises(ValueError,match='frozen'):study.train_stage(ready,'tuning')


def test_review_receipt_is_exact_catalog_bound_and_complete():
    good={'schema':'l1-catalog-human-review-v1','catalog_sha256':'a'*64,'decision':'approved','reviewer':'human','reviewed_at':'2026-10-04T00:00:00Z','forms_reviewed':48}
    assert study.validate_review(good,'a'*64)['decision']=='approved'
    for change in ({'catalog_sha256':'b'*64},{'decision':'pending'},{'reviewer':''},{'forms_reviewed':47}):
        with pytest.raises(ValueError):study.validate_review(good|change,'a'*64)


def test_source_or_dataset_change_invalidates_existing_lock(ready):
    lock=json.loads((ready/'study.json').read_text())
    path=Path(lock['dataset'])/'public/train.jsonl'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):study.status(ready)


def test_test_open_marker_prevents_further_training(ready):
    study.atomic_json(ready/'test-open.json',{'opened':True})
    with pytest.raises(ValueError,match='test'):study.train_stage(ready,'tuning')


def test_registered_scope_rejects_fixture_size_before_any_job(ready,tmp_path):
    lock=json.loads((ready/'study.json').read_text())
    with pytest.raises(ValueError,match='registered'):
        study.initialize(lock['dataset'],lock['cache'],tmp_path/'bad',code_revision='0'*40)


def test_completed_epoch_without_completion_stamp_is_resumable(ready):
    study.train_stage(ready,'tuning',max_jobs=1)
    run=next((ready/'jobs/tuning/concat').iterdir())
    before=json.loads((run/'report.json').read_text())['committed_optimizer_steps']
    (run/'completion.json').unlink()
    assert study.status(ready)['tuning_completed']==0
    study.train_stage(ready,'tuning',max_jobs=1)
    assert json.loads((run/'report.json').read_text())['committed_optimizer_steps']==before
    assert (run/'completion.json').exists()
