"""Epoch-transactional L1 training. Completion is never an L1 gate certificate."""
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import os
import platform
import resource
import sys
import numpy as np
import torch
from bcs.grounder import Grounder, ReaderKind
from bcs.language_tokens import PAD
from bcs.language_training import TrainConfig, _batch, validation_metrics
from bcs.sonar_encoder import file_hash


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def source_hashes():
    import bcs
    return {**{'bcs/'+p.name:file_hash(p) for p in sorted(Path(bcs.__file__).parent.glob('*.py'))},
            'l1/training.py':file_hash(Path(__file__))}


def _versions():
    return {'torch':str(torch.__version__),'numpy':np.__version__,'python':platform.python_version()}


def _context(kind,train,targets,validation,val_targets,config,identity,device):
    return {'schema':'l1-job-context-v1','reader':kind.value,'input_dim':train.vectors.shape[1],
            'identity':identity,'config':asdict(config),'device':str(device),'threads':torch.get_num_threads(),
            'versions':_versions(),'source_sha256':source_hashes(),
            'train_cache':train.manifest,'validation_cache':validation.manifest,
            'train_targets_sha256':digest(targets),'validation_targets_sha256':digest(val_targets)}


def _cpu_weights(model):
    return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


def _flops(kind,dimension,length,tokens):
    h=256
    if kind==ReaderKind.CONCAT:reader=2*((16*dimension+17)*h+h*h)
    elif kind==ReaderKind.GRU:reader=6*length*(dimension*h+h*h)
    else:reader=2*length*dimension*h+2*(8*length*h*h+4*length*length*h+4*length*h*1024)
    from bcs.language_tokens import VOCAB_SIZE
    decoder=tokens*(6*((64+h)*h+h*h)+2*h*VOCAB_SIZE)
    if kind==ReaderKind.ATTENTION:decoder+=tokens*(4*h*h+4*length*h+4*length*h*h)
    return {'per_history':reader+decoder,'mean_history_length':length,'decoder_tokens':tokens,
            'assumptions':'Analytical dense inference estimate, multiply-add=2; attention at unpadded mean length, decoder greedy length approximated by target length. Excludes embedding lookup, bias, nonlinearities, masks, framework overhead and backpropagation; not a profiler measurement.'}


def _resources(model,kind,cache,targets,device,elapsed,validation_seconds):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'process_peak_rss_bytes':int(rss if sys.platform=='darwin' else rss*1024),
            'rss_scope':'process lifetime high-water mark; not isolated model allocation',
            'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else None,
            'committed_wall_seconds':elapsed,'last_validation_seconds':validation_seconds,
            'validation_scope':'greedy inference plus teacher-forced likelihood on validation; includes batching overhead',
            'inference_flops_estimate':_flops(kind,model.input_dim,float(np.mean(list(map(len,cache.indices)))),float(np.mean(list(map(len,targets)))))}


def _validate_report(output,identity):
    report=json.loads((output/'report.json').read_text())
    context=json.loads((output/'context.json').read_text())
    if report['status']!='completed' or not (output/'completion.json').exists():raise ValueError('training run is not completed')
    completion=json.loads((output/'completion.json').read_text())
    if completion!={'report_sha256':file_hash(output/'report.json'),'context_sha256':digest(context),'weights_sha256':file_hash(output/'best.pt')}:raise ValueError('completed report/checkpoint changed')
    if (context['identity']!=identity or context['source_sha256']!=source_hashes()
        or report['context_sha256']!=digest(context) or report['weights_sha256']!=file_hash(output/'best.pt')):
        raise ValueError('completed checkpoint identity/checksum mismatch')
    return report,context


def load_completed_run(output,expected_identity,device='cpu'):
    output=Path(output);report,context=_validate_report(output,expected_identity)
    model=Grounder(ReaderKind(context['reader']),context['input_dim']).to(device)
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True))
    return model.eval()


def fit_run(kind,train_cache,train_targets,validation_cache,validation_targets,output,
            config:TrainConfig,identity:dict,device='cpu',max_new_epochs=None,progress=lambda row:None):
    """Resume only a matching immutable job; checkpoint at completed epoch boundaries.

    An interrupted, uncommitted epoch is replayed, so committed_optimizer_steps
    excludes abandoned work. max_new_epochs pauses without changing the budget.
    One process may own an output directory; concurrent writers are rejected.
    """
    output=Path(output);kind=ReaderKind(kind)
    if max_new_epochs is not None and (type(max_new_epochs)is not int or max_new_epochs<1):raise ValueError('positive epoch slice required')
    if set(train_cache.group_ids)&set(validation_cache.group_ids):raise ValueError('train/validation group overlap')
    if train_cache.manifest['encoder']!=validation_cache.manifest['encoder'] or train_cache.vectors.shape[1]!=validation_cache.vectors.shape[1]:raise ValueError('encoder identity mismatch')
    if not train_targets or not validation_targets or len(train_targets)!=len(train_cache.record_ids) or len(validation_targets)!=len(validation_cache.record_ids):raise ValueError('targets/cache alignment mismatch')
    context=_context(kind,train_cache,train_targets,validation_cache,validation_targets,config,identity,device)
    output.mkdir(parents=True,exist_ok=True)
    # Advisory OS lock is released on process death; it never needs stale-lock deletion.
    import fcntl
    with (output/'.writer.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('training output has an active writer') from error
        return _fit_locked(kind,train_cache,train_targets,validation_cache,validation_targets,output,config,identity,device,max_new_epochs,progress,context)


def _fit_locked(kind,train_cache,train_targets,validation_cache,validation_targets,output,config,identity,device,max_new_epochs,progress,context):
    if (output/'context.json').exists():
        if json.loads((output/'context.json').read_text())!=context:raise ValueError('resume context changed: inputs/configuration/code/runtime/identity')
    else:
        if any(p.name!='.writer.lock' for p in output.iterdir()):raise ValueError('unrecognized nonempty training destination')
        atomic_json(output/'context.json',context)
    if (output/'completion.json').exists():
        return _validate_report(output,identity)[0]
    torch.manual_seed(config.seed);torch.use_deterministic_algorithms(True)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(config.seed);torch.cuda.reset_peak_memory_stats(device)
    rng=np.random.default_rng(config.seed)
    model=Grounder(kind,context['input_dim']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.01)
    best=(float('inf'),float('inf'));best_state={};best_epoch=stale=steps=completed=0
    history=[];elapsed=0.;validation_seconds=0.;previous_checkpoint=None
    if (output/'resume.json').exists():
        pointer=json.loads((output/'resume.json').read_text());path=output/pointer['file']
        if path.parent!=output or pointer['context_sha256']!=digest(context) or file_hash(path)!=pointer['sha256']:raise ValueError('resume checkpoint checksum/identity mismatch')
        state=torch.load(path,map_location=device,weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        best=tuple(state['best']);best_state=state['best_state'];best_epoch=state['best_epoch'];stale=state['stale']
        steps=state['steps'];completed=state['completed'];history=state['history'];elapsed=state['elapsed'];validation_seconds=state['validation_seconds']
        rng.bit_generator.state=json.loads(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if str(device).startswith('cuda'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        previous_checkpoint=path
    end=config.epochs if max_new_epochs is None else min(config.epochs,completed+max_new_epochs)
    for epoch in range(completed+1,end+1):
        if stale>=config.patience:break
        started=perf_counter();model.train();loss_sum=count=0;order=rng.permutation(len(train_targets))
        for start in range(0,len(order),config.batch_size):
            z,n,t=_batch(train_cache,train_targets,order[start:start+config.batch_size],device)
            optimizer.zero_grad(set_to_none=True);logits=model.teacher_logits(z,n,t)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),t.flatten(),ignore_index=PAD)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();steps+=1
            count_here=int((t!=PAD).sum());loss_sum+=float(loss.detach())*count_here;count+=count_here
        validation_started=perf_counter()
        metrics=validation_metrics(model,validation_cache,validation_targets,config.batch_size,device)
        validation_seconds=perf_counter()-validation_started
        row={'epoch':epoch,'train_nll':loss_sum/count,'validation_event_error':metrics['full_event_error'],
             'validation_history_error':metrics['full_history_error'],'validation_nll':metrics['teacher_forced_nll']}
        history.append(row);score=(metrics['full_event_error'],metrics['teacher_forced_nll'])
        stale=0 if score[0]<best[0] else stale+1
        if score<best:best=score;best_epoch=epoch;best_state=_cpu_weights(model)
        completed=epoch;elapsed+=perf_counter()-started
        state={'model':_cpu_weights(model),'optimizer':optimizer.state_dict(),'best':list(best),'best_state':best_state,
               'best_epoch':best_epoch,'stale':stale,'steps':steps,'completed':completed,'history':history,'elapsed':elapsed,
               'validation_seconds':validation_seconds,'numpy_rng':json.dumps(rng.bit_generator.state),
               'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []}
        path=output/f'resume-epoch-{epoch:04d}.pt';temporary=path.with_suffix('.tmp')
        torch.save(state,temporary);os.replace(temporary,path)
        atomic_json(output/'resume.json',{'file':path.name,'sha256':file_hash(path),'context_sha256':digest(context)})
        if previous_checkpoint and previous_checkpoint!=path:previous_checkpoint.unlink(missing_ok=True)
        previous_checkpoint=path;progress(row)
    done=completed==config.epochs or stale>=config.patience
    weights_hash=None
    if done:
        torch.save(best_state,output/'best.tmp');os.replace(output/'best.tmp',output/'best.pt');weights_hash=file_hash(output/'best.pt')
    report={'schema':'l1-training-job-v1','status':'completed' if done else 'paused','L1_status':'not_evaluated',
            'context_sha256':digest(context),'reader':kind.value,'config':asdict(config),'dtype':'float32','device':str(device),
            'completed_epochs':completed,'committed_optimizer_steps':steps,'history':history,'best_epoch':best_epoch,
            'best_validation_event_error':best[0],'best_validation_nll':best[1],'weights_sha256':weights_hash,
            'stop_reason':'patience' if stale>=config.patience else 'epoch_budget' if done else 'epoch_slice',
            'parameters':sum(p.numel() for p in model.parameters()),
            'resources':_resources(model,kind,train_cache,train_targets,device,elapsed,validation_seconds)}
    atomic_json(output/'report.json',report)
    if done:atomic_json(output/'completion.json',{'report_sha256':file_hash(output/'report.json'),'context_sha256':digest(context),'weights_sha256':weights_hash})
    return report
