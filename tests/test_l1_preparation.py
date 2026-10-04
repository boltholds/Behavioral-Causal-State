import json
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from bcs.embedding_cache import load_cache


def module():
    path=Path(__file__).parents[1]/'experiments/l1/preparation.py'
    assert path.exists(), 'L1 preparation entrypoint is missing'
    spec=importlib.util.spec_from_file_location('l1_preparation',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class Encoder:
    dimension=2
    identity={'family':'fixture','normalization':'none'}
    def __init__(self):self.seen=[]
    def encode(self,texts):
        self.seen.extend(texts)
        return np.array([[len(t),sum(t.encode())] for t in texts],dtype=np.float32)


def sources(tmp_path):
    root=tmp_path/'dataset';(root/'public').mkdir(parents=True)
    for split,texts in zip(('train','validation','iid_test','challenge'),(('a','bb','a'),('bb','ccc'),('a','d'),('d','eee'))):
        row={'record_id':split+':0','group_id':split,'messages':[{'role':'user','speaker_id':'speaker-0','text':t} for t in texts]}
        (root/'public'/f'{split}.jsonl').write_text(json.dumps(row)+'\n')
    return root


def test_deduplicates_across_splits_without_changing_order(tmp_path):
    prep=module();dataset=sources(tmp_path);encoder=Encoder();output=tmp_path/'cache'
    result=prep.prepare(dataset,encoder,output,batch_size=1)
    assert encoder.seen==['a','bb','ccc','d','eee']
    assert result['global_unique_sentences']==5
    cache=load_cache(output/'train',dataset/'public/train.jsonl')
    assert cache.batch([0])[0][0,:3,0].tolist()==[1,2,1]
    assert result['splits']['train']['records']==1


def test_verified_resume_reuses_vectors_and_refuses_corruption(tmp_path):
    prep=module();dataset=sources(tmp_path);output=tmp_path/'cache'
    prep.prepare(dataset,Encoder(),output)
    encoder=Encoder();result=prep.prepare(dataset,encoder,output)
    assert encoder.seen==[] and all(s['resumed'] for s in result['splits'].values())
    with (output/'challenge/vectors.npy').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='checksum'):prep.prepare(dataset,Encoder(),output)


def test_partial_output_is_never_reused(tmp_path):
    prep=module();dataset=sources(tmp_path);output=tmp_path/'cache'
    (output/'.train.partial').mkdir(parents=True)
    encoder=Encoder()
    with pytest.raises(ValueError,match='partial'):prep.prepare(dataset,encoder,output)
    assert encoder.seen==[]


def test_labels_and_changed_identity_are_rejected(tmp_path):
    prep=module();dataset=sources(tmp_path);output=tmp_path/'cache'
    prep.prepare(dataset,Encoder(),output)
    encoder=Encoder();encoder.identity={'family':'changed'}
    with pytest.raises(ValueError,match='identity'):prep.prepare(dataset,encoder,output)
    path=dataset/'public/train.jsonl';row=json.loads(path.read_text());row['labels']=[1];path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):prep.prepare(dataset,Encoder(),tmp_path/'other')


def test_completed_splits_seed_remaining_split_without_reencoding(tmp_path):
    import shutil
    prep=module();dataset=sources(tmp_path);output=tmp_path/'cache'
    prep.prepare(dataset,Encoder(),output)
    shutil.rmtree(output/'challenge')
    encoder=Encoder();result=prep.prepare(dataset,encoder,output)
    assert encoder.seen==['eee']
    assert result['splits']['challenge']['newly_encoded']==1
    assert result['global_unique_sentences']==5
