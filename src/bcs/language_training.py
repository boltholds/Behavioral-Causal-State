"""Reproducible single-configuration diagnostic training; L1 gates stay not_run."""
from dataclasses import dataclass,asdict
from pathlib import Path
from time import perf_counter
from hashlib import sha256
import json
import math
import platform
import numpy as np
import torch
from .grounder import Grounder,ReaderKind
from .dataset_codec import event_from_wire
from .language_tokens import encode_events,PAD
from .language_evaluation import structural_report
from .sonar_encoder import file_hash


@dataclass(frozen=True)
class TrainConfig:
    epochs: int=30
    batch_size: int=64
    learning_rate: float=.0003
    seed: int=11
    patience: int=5
    def __post_init__(self):
        if any(type(v) is not int for v in (self.epochs,self.batch_size,self.seed,self.patience)) or not 1<=self.epochs<=30 or self.batch_size<1 or self.seed<0 or not 1<=self.patience<=5 or not math.isfinite(self.learning_rate) or self.learning_rate<=0:
            raise ValueError('invalid training configuration')


def load_labels(path,cache):
    labels={}
    with Path(path).open() as file:
        for line in file:
            row=json.loads(line)
            if set(row)!={'record_id','events'} or row['record_id'] in labels:
                raise ValueError('invalid/duplicate training label row')
            labels[row['record_id']]=encode_events(tuple(event_from_wire(e) for e in row['events']))
    if set(labels)!=set(cache.record_ids):
        raise ValueError('label IDs must exactly match this public split')
    return tuple(labels[rid] for rid in cache.record_ids)


def _batch(cache,targets,indices,device):
    z,lengths=cache.batch(indices)
    selected=[targets[i] for i in indices]
    padded=torch.full((len(indices),max(map(len,selected))),PAD,dtype=torch.long,device=device)
    for i,tokens in enumerate(selected):padded[i,:len(tokens)]=torch.tensor(tokens,device=device)
    return torch.from_numpy(z).to(device),torch.from_numpy(lengths).to(device),padded


@torch.inference_mode()
def predict_cache(model,cache,batch_size=64,device='cpu'):
    model.eval();predictions=[]
    for start in range(0,len(cache.record_ids),batch_size):
        z,n=cache.batch(range(start,min(start+batch_size,len(cache.record_ids))))
        predictions.extend(model.greedy(torch.from_numpy(z).to(device),torch.from_numpy(n).to(device)))
    return tuple(predictions)


@torch.inference_mode()
def validation_metrics(model,cache,targets,batch_size,device):
    predictions=predict_cache(model,cache,batch_size,device)
    report=structural_report(predictions,targets);loss_sum=count=0
    for start in range(0,len(targets),batch_size):
        z,n,t=_batch(cache,targets,list(range(start,min(start+batch_size,len(targets)))),device)
        logits=model.teacher_logits(z,n,t)
        loss_sum+=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),t.flatten(),ignore_index=PAD,reduction='sum'))
        count+=int((t!=PAD).sum())
    report['teacher_forced_nll']=loss_sum/count
    return report


def train_one(kind,train_cache,train_targets,validation_cache,validation_targets,output,config=TrainConfig(),device='cpu',progress=lambda value:None,*,input_lock_sha256):
    output=Path(output)
    if len(input_lock_sha256)!=64 or any(c not in '0123456789abcdef' for c in input_lock_sha256):
        raise ValueError('verified input lock identity required')
    if output.exists() and any(output.iterdir()):raise FileExistsError('training destination must be empty')
    if set(train_cache.group_ids)&set(validation_cache.group_ids):raise ValueError('train/validation group overlap')
    if train_cache.manifest['encoder']!=validation_cache.manifest['encoder'] or train_cache.vectors.shape[1]!=validation_cache.vectors.shape[1]:raise ValueError('encoder identity mismatch')
    if len(train_targets)!=len(train_cache.record_ids) or len(validation_targets)!=len(validation_cache.record_ids):raise ValueError('targets/cache alignment mismatch')
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(config.seed)
    model=Grounder(kind,train_cache.vectors.shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=.01)
    best=(float('inf'),float('inf'));best_state={};best_epoch=0;stale=0;history=[];steps=0
    started=perf_counter()
    for epoch in range(1,config.epochs+1):
        model.train();loss_sum=count=0;order=rng.permutation(len(train_targets))
        for start in range(0,len(order),config.batch_size):
            z,n,t=_batch(train_cache,train_targets,order[start:start+config.batch_size],device)
            optimizer.zero_grad(set_to_none=True)
            logits=model.teacher_logits(z,n,t)
            loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),t.flatten(),ignore_index=PAD)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();steps+=1
            tokens=int((t!=PAD).sum());loss_sum+=float(loss.detach())*tokens;count+=tokens
        validation=validation_metrics(model,validation_cache,validation_targets,config.batch_size,device)
        row={'epoch':epoch,'train_nll':loss_sum/count,'validation_event_error':validation['full_event_error'],'validation_history_error':validation['full_history_error'],'validation_nll':validation['teacher_forced_nll']}
        history.append(row);progress(row)
        score=(validation['full_event_error'],validation['teacher_forced_nll'])
        if score<best:
            best=score;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=config.patience:break
    model.load_state_dict(best_state);model.eval()
    validation=validation_metrics(model,validation_cache,validation_targets,config.batch_size,device)
    torch.save(best_state,output/'model.pt')
    provenance={'schema':'grounder-checkpoint-v1','reader':kind.value,'input_dim':model.input_dim,'encoder':train_cache.manifest['encoder'],
                'train_cache':train_cache.manifest,
                'validation_cache':validation_cache.manifest,'weights_sha256':file_hash(output/'model.pt'),
                'train_targets_sha256':sha256(json.dumps(train_targets,separators=(',',':')).encode()).hexdigest(),
                'validation_targets_sha256':sha256(json.dumps(validation_targets,separators=(',',':')).encode()).hexdigest(),
                'config':asdict(config),'best_epoch':best_epoch,'protocol_scope':'diagnostic',
                'input_lock_sha256':input_lock_sha256,
                'source_sha256':{p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    (output/'checkpoint.json').write_text(json.dumps(provenance,indent=2)+'\n')
    report={'schema':'language-training-v1','L1_status':'not_run_diagnostic','reader':kind.value,'settings':asdict(config),
            'train_histories':len(train_targets),'validation_histories':len(validation_targets),'history':history,'best_epoch':best_epoch,
            'validation':validation,'parameters':sum(p.numel() for p in model.parameters()),'optimizer_steps':steps,'wall_seconds':perf_counter()-started,
            'device':device,'dtype':'float32','versions':{'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version()},
            'checkpoint_manifest_sha256':file_hash(output/'checkpoint.json'),'geometry_diagnostic':'deferred','heldout_test_metrics':'not_run',
            'full_protocol_blockers':['catalog human review','three learning-rate trials','five final seeds','heldout evaluation and registered gates']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def load_checkpoint(output,encoder_identity,expected_input_lock_sha256,device='cpu'):
    output=Path(output)
    metadata=json.loads((output/'checkpoint.json').read_text())
    report=json.loads((output/'report.json').read_text())
    from .language_lock import source_hashes
    if metadata['schema']!='grounder-checkpoint-v1' or metadata['encoder']!=encoder_identity or metadata['weights_sha256']!=file_hash(output/'model.pt') or metadata['protocol_scope']!='diagnostic' or metadata['input_lock_sha256']!=expected_input_lock_sha256 or metadata['source_sha256']!=source_hashes() or report['checkpoint_manifest_sha256']!=file_hash(output/'checkpoint.json'):
        raise ValueError('checkpoint identity/checksum mismatch')
    model=Grounder(ReaderKind(metadata['reader']),metadata['input_dim']).to(device)
    model.load_state_dict(torch.load(output/'model.pt',map_location=device,weights_only=True))
    return model.eval()
