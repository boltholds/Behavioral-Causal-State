"""Real 24-job pipeline exercise with synthetic features; never an L1 result."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch
from bcs.dataset import build_dataset,DatasetCounts
from bcs.sonar_encoder import file_hash
from .preparation import prepare
from .study import initialize,train_stage,freeze_selection,evaluate_study
from .training import atomic_json


class FixtureEncoder:
    dimension=4
    identity={'family':'sha256-fixture','causal_semantics':False}
    def encode(self,texts):
        return np.array([list(sha256(text.encode()).digest()[:4]) for text in texts],dtype=np.float32)/255


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args();root=args.output.resolve()
    if root.exists() and any(root.iterdir()):raise ValueError('fresh fixture output required')
    torch.set_num_threads(1)
    dataset=root/'dataset';cache=root/'cache';study=root/'study'
    build_dataset(dataset,DatasetCounts(4,4,4,2),seed=114)
    prepare(dataset,FixtureEncoder(),cache)
    initialize(dataset,cache,study,code_revision='0'*40,scope='fixture',fixture_epochs=1)
    tuning=train_stage(study,'tuning');selection=freeze_selection(study);final=train_stage(study,'final')
    result=evaluate_study(study)
    # Exercise completed-read path and deterministic evaluation replay without refit.
    assert evaluate_study(study)==result
    summary={'scope':'integration_fixture_not_L1','features':'4-dimensional SHA256 fixture; no SONAR quality inference',
             'counts':{'train':4,'validation':4,'iid':4,'challenge_pairs_per_stratum':2},
             'epochs_per_job':1,'tuning':tuning,'final':final,'selection':selection['branches'],'result':result,
             'study_lock':json.loads((study/'study.json').read_text()),
             'source_sha256':file_hash(Path(__file__)),
             'job_reports':{str(p.parent.relative_to(study/'jobs')):json.loads(p.read_text()) for p in sorted((study/'jobs').rglob('report.json'))}}
    atomic_json(args.report,summary)
    print(json.dumps({'tuning':tuning,'final':final,'L1_status':result['L1_status']}))


if __name__=='__main__':main()
