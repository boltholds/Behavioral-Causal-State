import json
from pathlib import Path
from dataclasses import replace
import pytest
from bcs.dataset import DatasetCounts, build_dataset, audit_dataset, semantic_fingerprint
from bcs.pairs import make_pair,PairKind,HoldoutStratum


def small_counts():
    return DatasetCounts(20,8,8,2)


def test_build_is_reproducible_and_public_records_cannot_contain_test_labels(tmp_path):
    a,b=tmp_path/'a',tmp_path/'b'
    build_dataset(a,small_counts(),seed=51)
    build_dataset(b,small_counts(),seed=51)
    for path in sorted(a.rglob('*.jsonl')):
        assert path.read_bytes() == (b/path.relative_to(a)).read_bytes()
    report=audit_dataset(a)
    assert report['integrity_status'] == 'passed'
    assert report['base_counts'] == {'train':20,'validation':8,'iid_test':8}
    assert report['challenge_groups'] == 28
    assert report['cross_split_semantic_duplicates'] == 0
    assert report['training_ready'] is False  # human catalog review/checkpoint lock pending
    for path in (a/'public').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            assert set(json.loads(line)) == {'record_id','group_id','messages'}


def test_time_permutation_has_same_semantic_fingerprint():
    pair=make_pair(4,'key',PairKind.TIME,HoldoutStratum.TEMPLATES)
    assert semantic_fingerprint(pair.left.events) == semantic_fingerprint(pair.right.events)

@pytest.mark.parametrize('fault',['public_label','private_label','missing_record','wrong_witness','split_duplicate'])
def test_audit_rejects_real_artifact_corruption(tmp_path,fault):
    build_dataset(tmp_path,small_counts(),seed=17)
    if fault == 'public_label':
        p=tmp_path/'public/iid_test.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        rows[0]['labels']=[1]
    elif fault == 'private_label':
        p=tmp_path/'evaluator/iid_test.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        rows[0]['sides'][0]['events'][0]['value']['value'] ^= 1
    elif fault == 'missing_record':
        p=tmp_path/'public/train.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()][1:]
    elif fault == 'wrong_witness':
        p=tmp_path/'evaluator/challenge.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        next(row for row in rows if row['kind']=='order')['annotation']['tv']='0'
    else:
        p=tmp_path/'evaluator/iid_test.jsonl'
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        train=json.loads((tmp_path/'evaluator/train.jsonl').read_text().splitlines()[0])
        rows[0]['sides'][0]['events']=train['sides'][0]['events']
    p.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    # Repair the checksum deliberately: semantic validation must catch the fault.
    from hashlib import sha256
    manifest_path=tmp_path/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['files'][str(p.relative_to(tmp_path))]=sha256(p.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    report=audit_dataset(tmp_path)
    assert report['integrity_status'] == 'failed'
    assert report['errors']


def test_nonempty_destination_is_not_overwritten(tmp_path):
    (tmp_path/'keep.txt').write_text('user content')
    with pytest.raises(FileExistsError):
        build_dataset(tmp_path,small_counts(),seed=0)
    assert (tmp_path/'keep.txt').read_text()=='user content'


def _repair_checksum(root,path):
    from hashlib import sha256
    mp=root/'manifest.json'; manifest=json.loads(mp.read_text())
    manifest['files'][str(path.relative_to(root))]=sha256(path.read_bytes()).hexdigest()
    mp.write_text(json.dumps(manifest))

@pytest.mark.parametrize('fault',['false_evidence','joint_preparation'])
def test_audit_rejects_forged_calibration_even_if_core_unchanged(tmp_path,fault):
    build_dataset(tmp_path,DatasetCounts(2,2,2,1),seed=3)
    path=tmp_path/'calibration.json'; panel=json.loads(path.read_text())
    if fault=='false_evidence':
        panel['probes'][0]['history']['evidence'].append({'time':0,'object_id':0,'variable':'R','value':1})
    else:
        panel['probes'][0]['history']['steps'][0]['clamps'].append({'object_id':0,'variable':'C','value':1})
    path.write_text(json.dumps(panel)); _repair_checksum(tmp_path,path)
    assert audit_dataset(tmp_path)['integrity_status']=='failed'


def test_audit_rejects_length_five_with_consistent_text_and_labels(tmp_path):
    from bcs.dataset_codec import event_from_wire
    from bcs.generator import Unknown,event_wire
    from bcs.simulator import Variable as V
    from bcs.language_catalog import render_catalog
    from dataclasses import asdict
    build_dataset(tmp_path,DatasetCounts(2,8,2,1),seed=3)
    private=tmp_path/'evaluator/validation.jsonl'; public=tmp_path/'public/validation.jsonl'
    rows=[json.loads(s) for s in private.read_text().splitlines()]
    texts=[json.loads(s) for s in public.read_text().splitlines()]
    index=next(i for i,r in enumerate(rows) if len(r['sides'][0]['events']) in (4,8,12))
    row=rows[index]; events=tuple(event_from_wire(e) for e in row['sides'][0]['events'])
    events=events+(Unknown(events[-1].time,0,V.R),)
    row['sides'][0]['events']=[event_wire(e) for e in events]
    texts[index]['messages']=[asdict(m) for m in render_catalog(events,tuple(row['names']),row['families'][0],row['explicit_time'])]
    for p,data in ((private,rows),(public,texts)):
        p.write_text(''.join(json.dumps(r)+'\n' for r in data));_repair_checksum(tmp_path,p)
    assert audit_dataset(tmp_path)['integrity_status']=='failed'


def test_extra_public_sidecar_is_not_silently_safe(tmp_path):
    build_dataset(tmp_path,DatasetCounts(2,2,2,1),seed=3)
    (tmp_path/'public/leaked_labels.jsonl').write_text('{"labels":[1]}\n')
    assert audit_dataset(tmp_path)['integrity_status']=='failed'
