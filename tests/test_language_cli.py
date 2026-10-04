import json
from pathlib import Path
import subprocess
import sys
import pytest
pytest.importorskip('torch')
from bcs.language_cli import validation_events
from bcs.dataset import build_dataset,DatasetCounts
from bcs.embedding_cache import build_cache,load_cache
import numpy as np


class Encoder:
    dimension=2;identity={'family':'test-fixture'}
    def encode(self,texts):return np.ones((len(texts),2),dtype=np.float32)


def test_validation_label_loader_rejects_other_split(tmp_path):
    build_dataset(tmp_path/'data',DatasetCounts(2,2,2,1),seed=8)
    public=tmp_path/'data/public/validation.jsonl'
    build_cache(public,Encoder(),tmp_path/'cache')
    cache=load_cache(tmp_path/'cache',public)
    assert len(validation_events(tmp_path/'data/evaluator/validation.jsonl',cache))==2
    with pytest.raises(ValueError):validation_events(tmp_path/'data/evaluator/train.jsonl',cache)


def test_cli_explains_diagnostic_scope_without_loading_sonar():
    result=subprocess.run([sys.executable,'-m','bcs.language_cli','train','--help'],capture_output=True,text=True)
    assert result.returncode==0
    assert 'diagnostic' in result.stdout


def test_cli_rejects_training_without_input_lock(tmp_path):
    dataset=tmp_path/'data';cache_root=tmp_path/'cache'
    build_dataset(dataset,DatasetCounts(2,2,2,1),seed=8)
    encoder=Encoder();encoder.identity={'family':'SONAR'}
    for split in ('train','validation'):
        build_cache(dataset/f'public/{split}.jsonl',encoder,cache_root/split)
    result=subprocess.run([sys.executable,'-m','bcs.language_cli','train','--dataset',str(dataset),
        '--cache-root',str(cache_root),'--branch','gru','--epochs','1','--output',str(tmp_path/'run')],capture_output=True,text=True)
    assert result.returncode!=0
