"""Locked L1 study: nine tuning jobs, frozen selection, fifteen seeds, then test.

Run from the source checkout with PYTHONPATH=src. Fixture scope is explicitly
ineligible for L1. A human review receipt is required for registered training.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import torch
from bcs.embedding_cache import load_cache
from bcs.language_training import TrainConfig,load_labels,predict_cache
from bcs.language_cli import validation_events
from bcs.language_tokens import encode_events
from bcs.sonar_encoder import file_hash,validate_sonar_identity
from .training import fit_run,load_completed_run,_validate_report,_versions,atomic_json,digest,source_hashes as training_sources
from .runtime import configure_runtime

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'experiments/protocols/language-v0.1.json'
BRANCHES=('concat','attention','gru')
SPLITS=('train','validation','iid_test','challenge')
PREFLIGHT_TESTS=('test_simulator.py','test_inference.py','test_contracts.py','test_population.py','test_journal.py','test_dataset.py','test_embedding_cache.py','test_language_cli.py')


def _read(path):return json.loads(Path(path).read_text())
def _now():return datetime.now(timezone.utc).isoformat()


def _sources():
    return training_sources()|{str(p.relative_to(ROOT)):file_hash(p) for p in sorted((ROOT/'experiments/l1').glob('*.py'))}|{'tests/'+n:file_hash(ROOT/'tests'/n) for n in PREFLIGHT_TESTS}


def validate_review(receipt,catalog_hash):
    if (receipt.get('schema')!='l1-catalog-human-review-v1' or receipt.get('catalog_sha256')!=catalog_hash
        or receipt.get('decision')!='approved' or not isinstance(receipt.get('reviewer'),str) or not receipt['reviewer'].strip()
        or receipt.get('forms_reviewed')!=48 or not receipt.get('reviewed_at')):
        raise ValueError('catalog human review must approve all 48 forms and match exact catalog hash')
    datetime.fromisoformat(receipt['reviewed_at'].replace('Z','+00:00'))
    return receipt


def initialize(dataset,cache,output,*,code_revision,scope='registered',review=None,device='cpu',fixture_epochs=None):
    from bcs.dataset import audit_dataset
    numeric_runtime=configure_runtime(device)
    dataset,cache,output=Path(dataset).resolve(),Path(cache).resolve(),Path(output).resolve()
    if scope not in ('registered','fixture'):raise ValueError('unknown study scope')
    if output.exists() and any(output.iterdir()):raise ValueError('study destination must be empty; use resume commands')
    manifest=_read(dataset/'manifest.json');protocol=_read(PROTOCOL)
    if not re.fullmatch('[0-9a-f]{40}',code_revision):raise ValueError('40-digit code revision required')
    if scope=='registered':
        if (manifest['mode']!='registered_size' or manifest['counts']!={'train':20000,'validation':4000,'iid_test':4000,'pairs_per_stratum':1000}
            or fixture_epochs is not None or code_revision=='0'*40):raise ValueError('registered study requires full sizes, real published revision and unchanged budgets')
    audit=audit_dataset(dataset)
    if audit['integrity_status']!='passed':raise ValueError('dataset audit failed: '+str(audit['errors']))
    if manifest.get('protocol_sha256',file_hash(PROTOCOL) if scope=='fixture' else None)!=file_hash(PROTOCOL):raise ValueError('dataset protocol mismatch')
    caches={s:load_cache(cache/s,dataset/f'public/{s}.jsonl') for s in SPLITS}
    identity=caches['train'].manifest['encoder']
    if any(c.manifest['encoder']!=identity for c in caches.values()):raise ValueError('encoder mismatch across splits')
    if scope=='registered':validate_sonar_identity(identity,caches['train'].vectors.shape[1])
    receipt=validate_review(_read(review),file_hash(dataset/'catalog.json')) if review else None
    training=dict(protocol['training'])
    if scope=='fixture':
        training['max_epochs']=fixture_epochs or 1
        TrainConfig(epochs=training['max_epochs'])
    lock={'schema':'l1-study-v1','scope':scope,'created_at':_now(),'code_commit':code_revision,
          'dataset':str(dataset),'cache':str(cache),'protocol_sha256':file_hash(PROTOCOL),
          'dataset_files':{n:file_hash(dataset/n) for n in ('manifest.json',*manifest['files'])},
          'cache_files':{f'{s}/{n}':file_hash(cache/s/n) for s in SPLITS for n in ('manifest.json','records.json','vectors.npy')},
          'source_sha256':_sources(),'training':training,'encoder':identity,'catalog_review':receipt,
          'device':device,'threads':torch.get_num_threads(),'runtime_versions':_versions(),'numeric_runtime':numeric_runtime,'audit':audit,'core_artifact_id':manifest['core_artifact_id'],
          'assumption_ids':['deterministic','known_graph','known_binary_variables','fixed_learned_boolean_mechanisms','perfect_clamps','observed_initial_C','fixed_data'],
          'heldout_features':'frozen encoder on public text only; no normalization/statistic fitting',
          'preflight':None}
    output.mkdir(parents=True,exist_ok=True)
    if scope=='registered':
        from experiments.ttt.runner import validate_report
        tttpath=ROOT/'experiments/results/ttt-v0.1/report.json'
        ttt=validate_report(_read(tttpath))
        command=[sys.executable,'-m','pytest','-q',*[str(ROOT/'tests'/n) for n in PREFLIGHT_TESTS]]
        run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        (output/'preflight.log').write_text(run.stdout+run.stderr)
        if run.returncode:raise ValueError('analytic/status/replay/leakage preflight failed; see preflight.log')
        lock['preflight']={'status':'passed','tests':list(PREFLIGHT_TESTS),'log_sha256':file_hash(output/'preflight.log'),
                           'ttt':ttt,'ttt_report_sha256':file_hash(tttpath)}
    atomic_json(output/'study.json',lock)
    return status(output)


def _lock(output):
    output=Path(output);lock=_read(output/'study.json')
    if lock['source_sha256']!=_sources() or lock['protocol_sha256']!=file_hash(PROTOCOL):raise ValueError('study source/protocol changed')
    if lock.get('numeric_runtime')!=configure_runtime(lock['device']):raise ValueError('locked numeric runtime/GPU changed')
    if lock['threads']!=torch.get_num_threads() or lock['runtime_versions']!=_versions():raise ValueError('locked runtime/thread count changed')
    if lock['scope']=='registered' and lock['training']!=_read(PROTOCOL)['training']:raise ValueError('registered training budget changed')
    for field,root in (('dataset_files',Path(lock['dataset'])),('cache_files',Path(lock['cache']))):
        for name,expected in lock[field].items():
            if not (root/name).is_file() or file_hash(root/name)!=expected:raise ValueError(f'locked input changed: {name}')
    if lock['scope']=='registered':
        evidence=lock['preflight']
        if not evidence or evidence['log_sha256']!=file_hash(output/'preflight.log'):raise ValueError('preflight evidence changed')
        path=ROOT/'experiments/results/ttt-v0.1/report.json'
        if evidence['ttt_report_sha256']!=file_hash(path):raise ValueError('TTT evidence changed')
        from experiments.ttt.runner import source_hashes as ttt_sources
        if _read(path)['manifest']['source_sha256']!=ttt_sources():raise ValueError('TTT source changed')
    return lock


def attach_review(output,receipt):
    output=Path(output);lock=_lock(output)
    if (output/'jobs').exists() or (output/'selection.json').exists() or (output/'test-open.json').exists():raise ValueError('review must precede all training')
    value=validate_review(_read(receipt),lock['dataset_files']['catalog.json'])
    if lock['catalog_review'] and lock['catalog_review']!=value:raise ValueError('review already frozen')
    lock['catalog_review']=value;atomic_json(output/'study.json',lock)
    return status(output)


def _guard_training(output,lock):
    if (output/'test-open.json').exists():raise ValueError('test is open: training and selection are closed')
    if lock['scope']=='registered' and not lock['catalog_review']:raise ValueError('catalog human review pending; attach a receipt before registered training')


def _jobs(lock,stage,selection=None):
    cfg=lock['training']
    if stage=='tuning':
        return [(f'tuning/{b}/lr-{lr:g}',b,lr,cfg['tuning_seed']) for b in BRANCHES for lr in cfg['learning_rates']]
    if stage!='final' or selection is None:raise ValueError('final jobs require frozen selection')
    return [(f'final/{b}/seed-{seed}',b,selection['branches'][b]['learning_rate'],seed) for b in BRANCHES for seed in cfg['final_seeds']]


def _identity(lock,key):return {'study_sha256':digest(lock),'job':key}


def _job_report(output,lock,key):
    path=output/'jobs'/key
    if not (path/'report.json').exists():return None
    report=_read(path/'report.json')
    if report['status']!='completed' or not (path/'completion.json').exists():return None
    report,context=_validate_report(path,_identity(lock,key))
    stage,branch,label=key.split('/')
    if stage=='tuning':
        lr=float(label.removeprefix('lr-'));seed=lock['training']['tuning_seed']
    else:
        # Read the selection only after its independently derived winner was checked by caller.
        lr=_read(output/'selection.json')['branches'][branch]['learning_rate'];seed=int(label.removeprefix('seed-'))
    expected=TrainConfig(epochs=lock['training']['max_epochs'],batch_size=lock['training']['batch_size'],learning_rate=lr,seed=seed,patience=lock['training']['early_stopping_patience'])
    from dataclasses import asdict
    if context['reader']!=branch or context['config']!=asdict(expected):raise ValueError('completed job configuration differs from frozen budget/selection')
    return report


def _choices(output,lock):
    jobs=_jobs(lock,'tuning');reports={key:_job_report(output,lock,key) for key,_,_,_ in jobs}
    if any(v is None for v in reports.values()):raise ValueError('all nine tuning jobs must be completed')
    choices={}
    for branch in BRANCHES:
        candidates=[(reports[key]['best_validation_event_error'],reports[key]['best_validation_nll'],lr,key) for key,b,lr,_ in jobs if b==branch]
        _,_,lr,key=min(candidates)
        choices[branch]={'learning_rate':lr,'tuning_job':key,'weights_sha256':reports[key]['weights_sha256']}
    return choices,{key:file_hash(output/'jobs'/key/'report.json') for key in reports}


def _selection(output,lock):
    if not (output/'selection.json').exists():raise ValueError('final jobs require frozen tuning selection')
    selection=_read(output/'selection.json')
    if selection['study_sha256']!=digest(lock):raise ValueError('selection study identity changed')
    choices,hashes=_choices(output,lock)
    if selection['branches']!=choices or selection['tuning_reports']!=hashes:raise ValueError('frozen tuning selection changed')
    return selection


def train_stage(output,stage,*,max_jobs=None,max_new_epochs=None):
    output=Path(output);lock=_lock(output);_guard_training(output,lock)
    if max_jobs is not None and (type(max_jobs)is not int or max_jobs<1):raise ValueError('positive job slice required')
    if stage=='tuning' and (output/'selection.json').exists():raise ValueError('tuning configuration frozen')
    selection=_selection(output,lock) if stage=='final' else None
    jobs=_jobs(lock,stage,selection);dataset,cache=Path(lock['dataset']),Path(lock['cache'])
    train=load_cache(cache/'train',dataset/'public/train.jsonl');val=load_cache(cache/'validation',dataset/'public/validation.jsonl')
    targets=load_labels(dataset/'training/labels.jsonl',train)
    val_targets=tuple(encode_events(e) for e in validation_events(dataset/'evaluator/validation.jsonl',val))
    budget=lock['training'];executed=0
    # No IID/challenge label reader is reachable from this training path.
    for key,branch,lr,seed in jobs:
        if _job_report(output,lock,key):continue
        result=fit_run(branch,train,targets,val,val_targets,output/'jobs'/key,
            TrainConfig(epochs=budget['max_epochs'],batch_size=budget['batch_size'],learning_rate=lr,seed=seed,patience=budget['early_stopping_patience']),
            _identity(lock,key),device=lock['device'],max_new_epochs=max_new_epochs,
            progress=lambda row:print(json.dumps({'job':key,**row}),flush=True))
        executed+=1
        if result['status']!='completed' or (max_jobs is not None and executed>=max_jobs):break
    return {'stage':stage,'completed_jobs':sum(_job_report(output,lock,key) is not None for key,_,_,_ in jobs),'total_jobs':len(jobs),'executed_jobs':executed}


def freeze_selection(output):
    output=Path(output);lock=_lock(output);_guard_training(output,lock)
    if (output/'selection.json').exists():return _selection(output,lock)
    choices,hashes=_choices(output,lock)
    result={'schema':'l1-selection-v1','study_sha256':digest(lock),'frozen_at':_now(),'branches':choices,
            'criterion':['validation_full_event_error','validation_teacher_forced_nll','lower_learning_rate'],
            'tuning_reports':hashes}
    atomic_json(output/'selection.json',result);return result


def _freeze_final(output,lock):
    selection=_selection(output,lock);jobs=_jobs(lock,'final',selection)
    if any(_job_report(output,lock,key) is None for key,_,_,_ in jobs):raise ValueError('all fifteen final jobs must be completed before test')
    value={'schema':'l1-final-freeze-v1','study_sha256':digest(lock),'selection_sha256':file_hash(output/'selection.json'),
           'checkpoints':{key:{'report_sha256':file_hash(output/'jobs'/key/'report.json'),'weights_sha256':file_hash(output/'jobs'/key/'best.pt')} for key,_,_,_ in jobs}}
    path=output/'final-freeze.json'
    if path.exists() and _read(path)!=value:raise ValueError('final frozen checkpoints changed')
    if not path.exists():atomic_json(path,value)
    return value


def evaluate_study(output):
    output=Path(output);lock=_lock(output)
    if not (output/'selection.json').exists():raise ValueError('final checkpoints must be frozen before test')
    frozen=_freeze_final(output,lock)
    marker={'study_sha256':digest(lock),'final_freeze_sha256':digest(frozen),'test_feedback_to_training':False}
    if (output/'test-open.json').exists():
        if _read(output/'test-open.json')!=marker:raise ValueError('test-open identity changed')
    else:
        _guard_training(output,lock);atomic_json(output/'test-open.json',marker)
    from .evaluation import evaluate_seed
    from time import perf_counter
    dataset,cache=Path(lock['dataset']),Path(lock['cache'])
    heldout={s:load_cache(cache/s,dataset/f'public/{s}.jsonl') for s in ('iid_test','challenge')}
    reports=[]
    for seed in lock['training']['final_seeds']:
        destination=output/'evaluation'/f'seed-{seed}';destination.mkdir(parents=True,exist_ok=True)
        stamp=destination/'complete.json'
        if stamp.exists():
            evidence=_read(stamp)
            if evidence['test_open']!=digest(marker) or evidence['report_sha256']!=file_hash(destination/'report.json') or evidence['predictions_sha256']!=file_hash(destination/'predictions.json'):raise ValueError('evaluation report changed')
            reports.append(_read(destination/'report.json'));continue
        predictions={};resources={}
        for branch in BRANCHES:
            key=f'final/{branch}/seed-{seed}'
            model=load_completed_run(output/'jobs'/key,_identity(lock,key),lock['device']);started=perf_counter();predictions[branch]={}
            for split,c in heldout.items():
                tokens=predict_cache(model,c,batch_size=lock['training']['batch_size'],device=lock['device'])
                predictions[branch].update(zip(c.record_ids,tokens))
            # Actual deterministic decoder replay, separate from journal persistence fixtures.
            c=heldout['iid_test'];n=min(64,len(c.record_ids));z,lengths=c.batch(range(n))
            replay=model.greedy(torch.from_numpy(z).to(lock['device']),torch.from_numpy(lengths).to(lock['device']))
            replay_ok=all(tuple(t)==tuple(predictions[branch][rid]) for rid,t in zip(c.record_ids[:n],replay))
            resources[branch]={'greedy_inference_seconds':perf_counter()-started,'histories':len(predictions[branch]),'replay_histories':n,'exact_token_replay':replay_ok}
            del model
        atomic_json(destination/'predictions.json',predictions)
        report=evaluate_seed(dataset,predictions,destination/'report.json');report['inference_resources']=resources
        report['study_sha256']=digest(lock);report['seed']=seed
        atomic_json(destination/'report.json',report)
        atomic_json(stamp,{'test_open':digest(marker),'report_sha256':file_hash(destination/'report.json'),'predictions_sha256':file_hash(destination/'predictions.json')})
        reports.append(report)
    numeric={b:sum(r['branches'][b]['gates']['numeric_combined']=='passed' and r['numeric_oracle_gate']=='passed' and r['inference_resources'][b]['exact_token_replay'] for r in reports) for b in BRANCHES}
    scope=lock['scope'];required=lock['training']['required_passing_seeds']
    # Rare-context estimand is missing in frozen v0.1; completion cannot conceal it.
    missing=sorted({r['rare_contexts']['status'] for r in reports if r['rare_contexts']['status']!='established'})
    numeric_status='not_run_fixture' if scope=='fixture' else ('passed' if all(n>=required for n in numeric.values()) else 'failed')
    result={'schema':'l1-study-result-v1','scope':scope,'study_sha256':digest(lock),'seeds':lock['training']['final_seeds'],
            'passing_seeds':numeric,'required_passing_seeds':required,'numeric_L1_status':numeric_status,
            'L1_status':'not_run_fixture' if scope=='fixture' else 'failed' if numeric_status=='failed' else 'incomplete_required_report' if missing else 'passed',
            'rare_context_status':'not_established' if missing else 'established',
            'preflight':lock['preflight'],'catalog_review':lock['catalog_review'],'final_freeze_sha256':digest(frozen),
            'seed_reports':{str(s):file_hash(output/'evaluation'/f'seed-{s}'/'report.json') for s in lock['training']['final_seeds']},
            'limitations':['Rare context projection/probability is unspecified in frozen v0.1.','Read-only dataset auditing may inspect evaluator sidecars; no held-out targets enter optimizer or model selection.']}
    atomic_json(output/'result.json',result)
    atomic_json(output/'result-complete.json',{'result_sha256':file_hash(output/'result.json')})
    return result


def status(output):
    output=Path(output);lock=_lock(output)
    if (output/'result-complete.json').exists():
        result=_read(output/'result.json');frozen=_freeze_final(output,lock)
        if _read(output/'result-complete.json')['result_sha256']!=file_hash(output/'result.json') or result['final_freeze_sha256']!=digest(frozen):raise ValueError('result changed')
        marker={'study_sha256':digest(lock),'final_freeze_sha256':digest(frozen),'test_feedback_to_training':False}
        if _read(output/'test-open.json')!=marker:raise ValueError('test marker changed')
        for seed,expected in result['seed_reports'].items():
            directory=output/'evaluation'/f'seed-{seed}';evidence=_read(directory/'complete.json')
            if expected!=file_hash(directory/'report.json') or evidence['report_sha256']!=expected or evidence['predictions_sha256']!=file_hash(directory/'predictions.json') or evidence['test_open']!=digest(marker):raise ValueError('seed report/predictions changed')
        return result
    return {'scope':lock['scope'],'L1_status':'not_run_fixture' if lock['scope']=='fixture' else 'blocked_catalog_review' if not lock['catalog_review'] else 'not_run',
            'tuning_completed':sum(_job_report(output,lock,key) is not None for key,_,_,_ in _jobs(lock,'tuning')),
            'selection_frozen':(output/'selection.json').exists(),'test_open':(output/'test-open.json').exists(),
            'registered_budget':{'tuning_jobs':9,'final_jobs':15,'max_epochs':lock['training']['max_epochs'],'batch_size':64}}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['init','review','status','tune','freeze','final','evaluate','run'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--dataset',type=Path);parser.add_argument('--cache',type=Path)
    parser.add_argument('--code-revision');parser.add_argument('--review',type=Path);parser.add_argument('--scope',choices=['registered','fixture'],default='registered')
    parser.add_argument('--fixture-epochs',type=int);parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--max-jobs',type=int);parser.add_argument('--max-new-epochs',type=int)
    args=parser.parse_args()
    if args.threads<1:parser.error('positive threads required')
    torch.set_num_threads(args.threads)
    # One process owns a study; job checkpoints also carry their own OS lock.
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output.parent/(args.output.name+'.study.lock')).open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:parser.error('study has an active writer')
        if args.command=='init':
            if not all((args.dataset,args.cache,args.code_revision)):parser.error('init requires dataset, cache, code-revision')
            result=initialize(args.dataset,args.cache,args.output,code_revision=args.code_revision,scope=args.scope,review=args.review,device=args.device,fixture_epochs=args.fixture_epochs)
        elif args.command=='review':
            if not args.review:parser.error('review requires --review receipt.json')
            result=attach_review(args.output,args.review)
        elif args.command=='status':result=status(args.output)
        elif args.command in ('tune','final'):result=train_stage(args.output,'tuning' if args.command=='tune' else 'final',max_jobs=args.max_jobs,max_new_epochs=args.max_new_epochs)
        elif args.command=='freeze':result=freeze_selection(args.output)
        elif args.command=='evaluate':result=evaluate_study(args.output)
        else:
            if args.max_jobs or args.max_new_epochs:parser.error('run uses full protocol; use tune/final for slices')
            if not (args.output/'test-open.json').exists():
                if not (args.output/'selection.json').exists():train_stage(args.output,'tuning');freeze_selection(args.output)
                train_stage(args.output,'final')
            result=evaluate_study(args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
