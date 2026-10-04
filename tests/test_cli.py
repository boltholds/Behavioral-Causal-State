import json
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    env = os.environ | {'PYTHONPATH':str(ROOT/'src')}
    return subprocess.run([sys.executable,'-m','bcs',*args],cwd=ROOT,env=env,capture_output=True,text=True)


def test_cli_demo_contains_exact_cf_and_no_false_discovery_claim():
    result = run('demo')
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output['counterfactual']['probabilities'] == [{'outcome':[0],'probability':'1'},{'outcome':[1],'probability':'0'}]
    assert output['scope'] == 'DeclaredFiniteClass'


def test_cli_g2_smoke_is_not_full_gate_success(tmp_path):
    path = tmp_path/'g2.json'
    result = run('g2','--smoke','--output',str(path))
    assert result.returncode == 0, result.stderr
    report = json.loads(path.read_text())
    assert report['acceptance_status'] == 'not_run_full_protocol'
    assert report['protocol_id'] == 'G2-MISSPEC-v0.1'
    assert len(report['manifest']['source_sha256']) >= 8
    assert report['controls']['undetectable_coupling']


def test_cli_generates_only_public_records_unless_train_labels_requested(tmp_path):
    path = tmp_path/'histories.jsonl'
    result = run('generate','--count','5','--regime','stochastic_partial','--output',str(path))
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 5
    assert all(set(record) == {'group_id','messages'} for record in records)
    assert path.with_suffix('.manifest.json').exists()
