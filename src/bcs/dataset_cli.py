"""CLI orchestration for reproducible calibration/data preparation."""
from pathlib import Path
from hashlib import sha256
import json
import sys
from .dataset import DatasetCounts,build_dataset,_dump
from .core_evaluation import evaluate_core


def _progress(value):
    print(json.dumps(value),file=sys.stderr,flush=True)


def prepare_l1(output,root,smoke=False,seed=20261004):
    root=Path(root); output=Path(output)
    protocol_path=root/'experiments/protocols/language-v0.1.json'
    protocol=json.loads(protocol_path.read_text())
    data=protocol['dataset']
    if data['new_template_pairs_per_challenge']!=data['new_names_compositions_pairs_per_challenge']:
        raise ValueError('balanced two-stratum protocol required')
    counts=DatasetCounts(20,8,8,2) if smoke else DatasetCounts(data['train_histories'],data['validation_histories'],data['iid_test_histories'],data['new_template_pairs_per_challenge'])
    manifest=build_dataset(output,counts,seed,_progress)
    manifest['protocol_sha256']=sha256(protocol_path.read_bytes()).hexdigest()
    _dump(output/'manifest.json',manifest)
    core=evaluate_core(output,_progress)
    audit=core['audit_report']
    _dump(output/'audit.json',audit)
    _dump(output/'core-evaluation.json',core)
    if audit['integrity_status']!='passed':
        return {'integrity_status':'failed','errors':audit['errors']}
    return {'integrity_status':audit['integrity_status'],'oracle_core_gate':core['oracle_validation_gate'],
            'training_ready':False,'output':str(output),'mode':manifest['mode']}
