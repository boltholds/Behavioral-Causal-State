import json
import numpy as np
import pytest
from bcs.dataset import build_dataset,DatasetCounts
from bcs.embedding_cache import build_cache
from bcs.sonar_encoder import pinned_identity
from bcs.language_lock import create_input_lock,verify_input_lock,prepare_subset

class Fixture:
    dimension=1024
    identity=pinned_identity()
    def encode(self,texts):return np.ones((len(texts),1024),dtype=np.float32)


def fixture(tmp_path):
    parent=tmp_path/'parent';dataset=tmp_path/'selected';cache=tmp_path/'cache'
    build_dataset(parent,DatasetCounts(4,4,2,1),seed=8)
    prepare_subset(parent,dataset,2,2)
    for split in ('train','validation'):build_cache(dataset/f'public/{split}.jsonl',Fixture(),cache/split)
    protocol=tmp_path/'protocol.json';protocol.write_text('{"test":"protocol"}')
    return parent,dataset,cache,protocol


@pytest.mark.parametrize('fault',['label','core','protocol','cache'])
def test_locked_inputs_reject_mutation_before_training(tmp_path,fault):
    parent,dataset,cache,protocol=fixture(tmp_path)
    lock_id=create_input_lock(dataset,parent,cache,protocol)
    assert verify_input_lock(dataset,cache,protocol)==lock_id
    path={'label':dataset/'training/labels.jsonl','core':dataset/'core.json',
          'protocol':protocol,'cache':cache/'train/manifest.json'}[fault]
    with path.open('a') as file:file.write(' ')
    with pytest.raises(ValueError):verify_input_lock(dataset,cache,protocol)


def test_cannot_lock_relabelled_subset_of_audited_parent(tmp_path):
    parent,dataset,cache,protocol=fixture(tmp_path)
    path=dataset/'training/labels.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
    event=next(e for e in rows[0]['events'] if e['mode']=='Executed')
    event['value']='stop' if event['value']=='start' else 'start'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):create_input_lock(dataset,parent,cache,protocol)
